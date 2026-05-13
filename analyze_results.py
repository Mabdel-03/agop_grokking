#!/usr/bin/env python
"""Aggregate AGOP grokking experiment outputs and compute lead-time tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import first_crossing
from src.plotting import make_all_plots
from src.utils import ensure_dir


def _markdown_table(df: pd.DataFrame) -> str:
    table = df.reset_index()
    cols = [str(c) for c in table.columns]
    rows = [[str(v) for v in row] for row in table.to_numpy()]
    widths = [len(c) for c in cols]
    for row in rows:
        widths = [max(w, len(v)) for w, v in zip(widths, row)]
    header = "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = ["| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |" for row in rows]
    return "\n".join([header, sep] + body)


def _read_many(results_dir: Path, name: str) -> pd.DataFrame:
    paths = sorted(results_dir.glob(f"**/{name}"))
    frames = []
    for path in paths:
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _transition_times(df: pd.DataFrame) -> tuple[int | None, int | None]:
    steps = df["step"].astype(int).tolist()
    t_train_fit = first_crossing(steps, df["train_acc"].tolist(), 0.99, "above")
    both = ((df["test_acc"] >= 0.95) & (df["train_acc"] >= 0.99)).astype(float).tolist()
    t_grok = first_crossing(steps, both, 0.5, "above", consecutive=2)
    return t_train_fit, t_grok


def _metric_at(df: pd.DataFrame, col: str, step: int | None) -> float:
    if step is None or col not in df:
        return float("nan")
    rows = df[df["step"] == step]
    if rows.empty:
        rows = df[df["step"] <= step].tail(1)
    if rows.empty:
        return float("nan")
    return float(rows[col].iloc[0])


def _relative_crossing(df: pd.DataFrame, col: str, frac: float, t_grok: int | None, decreasing: bool = False) -> int | None:
    vals = df[col].astype(float).copy()
    if vals.isna().all():
        return None
    vals = vals.ffill().bfill()
    start = float(vals.iloc[0])
    if t_grok is not None:
        end_rows = df[df["step"] <= t_grok]
        end = float(end_rows[col].dropna().iloc[-1]) if not end_rows[col].dropna().empty else float(vals.iloc[-1])
    else:
        end = float(vals.iloc[-1])
    if abs(end - start) < 1e-12:
        return None
    progress = (start - vals) / (start - end + 1e-12) if decreasing else (vals - start) / (end - start + 1e-12)
    return first_crossing(df["step"].astype(int).tolist(), progress.tolist(), frac, "above")


def _absolute_crossing(df: pd.DataFrame, col: str, threshold: float, direction: str) -> int | None:
    return first_crossing(df["step"].astype(int).tolist(), df[col].astype(float).tolist(), threshold, direction)


def _lead_row(
    *,
    run_id: str,
    seed: int,
    t_train_fit: int | None,
    t_grok: int | None,
    metric_name: str,
    threshold_type: str,
    threshold_value: float,
    t_metric: int | None,
    df: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "seed": int(seed),
        "t_train_fit": t_train_fit,
        "t_grok": t_grok,
        "metric_name": metric_name,
        "threshold_type": threshold_type,
        "threshold_value": threshold_value,
        "t_metric": t_metric,
        "lead_time_steps": (t_grok - t_metric) if t_grok is not None and t_metric is not None else np.nan,
        "lead_time_fraction_of_grok_time": ((t_grok - t_metric) / max(t_grok, 1)) if t_grok is not None and t_metric is not None else np.nan,
        "metric_value_at_t_metric": _metric_at(df, metric_name, t_metric),
        "metric_value_at_t_grok": _metric_at(df, metric_name, t_grok),
    }


def compute_lead_times(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame()
    metric_cols = []
    metric_cols.extend([c for c in metrics.columns if c.startswith("align_K")])
    metric_cols.extend([c for c in metrics.columns if c.startswith("random_align_K")])
    metric_cols.extend([c for c in metrics.columns if c.startswith("b_drift_")])
    metric_cols.extend([c for c in metrics.columns if c.startswith("g_drift_")])
    metric_cols.extend([c for c in metrics.columns if c.startswith("vcr_")])
    for col in ["log_weight_norm", "ntk_drift_correct_logit", "train_loss"]:
        if col in metrics.columns:
            metric_cols.append(col)
    metric_cols = list(dict.fromkeys(metric_cols))
    decreasing_set = {c for c in metric_cols if c.startswith("vcr_") or c == "train_loss"}

    for (run_id, seed), df in metrics.groupby(["run_id", "seed"], dropna=False):
        df = df.sort_values("step").reset_index(drop=True)
        t_train_fit, t_grok = _transition_times(df)
        for col in metric_cols:
            decreasing = col in decreasing_set
            for frac in (0.5, 0.8):
                rows.append(
                    _lead_row(
                        run_id=run_id,
                        seed=int(seed),
                        t_train_fit=t_train_fit,
                        t_grok=t_grok,
                        metric_name=col,
                        threshold_type=f"relative_{frac}",
                        threshold_value=frac,
                        t_metric=_relative_crossing(df, col, frac, t_grok, decreasing=decreasing),
                        df=df,
                    )
                )
            if col.startswith("align_K"):
                for thr in (0.2, 0.4, 0.6):
                    rows.append(
                        _lead_row(
                            run_id=run_id,
                            seed=int(seed),
                            t_train_fit=t_train_fit,
                            t_grok=t_grok,
                            metric_name=col,
                            threshold_type="absolute",
                            threshold_value=thr,
                            t_metric=_absolute_crossing(df, col, thr, "above"),
                            df=df,
                        )
                    )
            elif col == "ntk_drift_correct_logit":
                for thr in (0.25, 0.5, 1.0):
                    rows.append(
                        _lead_row(
                            run_id=run_id,
                            seed=int(seed),
                            t_train_fit=t_train_fit,
                            t_grok=t_grok,
                            metric_name=col,
                            threshold_type="absolute",
                            threshold_value=thr,
                            t_metric=_absolute_crossing(df, col, thr, "above"),
                            df=df,
                        )
                    )
            elif col == "train_loss":
                for thr in (0.05, 0.01):
                    rows.append(
                        _lead_row(
                            run_id=run_id,
                            seed=int(seed),
                            t_train_fit=t_train_fit,
                            t_grok=t_grok,
                            metric_name=col,
                            threshold_type="absolute",
                            threshold_value=thr,
                            t_metric=_absolute_crossing(df, col, thr, "below"),
                            df=df,
                        )
                    )
    return pd.DataFrame(rows)


def write_summary(results_dir: Path, metrics: pd.DataFrame, lead_times: pd.DataFrame) -> None:
    lines = ["# AGOP Grokking Summary", ""]
    if metrics.empty:
        lines.append("No metrics were found.")
        (results_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        return
    runs = metrics.groupby(["run_id", "seed"], dropna=False)
    grokked = []
    for (run_id, seed), df in runs:
        t_train_fit, t_grok = _transition_times(df.sort_values("step"))
        if t_train_fit is not None and t_grok is not None and t_grok > t_train_fit:
            grokked.append((run_id, seed, t_train_fit, t_grok))
    lines.append(f"- Runs analyzed: {len(runs)}")
    lines.append(f"- Grokking-like runs: {len(grokked)}")
    if grokked:
        lines.append("- Grokked seeds: " + ", ".join(f"seed {s} (fit {tf}, grok {tg})" for _, s, tf, tg in grokked))
    else:
        lines.append("- No run met the strict train-fit plus sustained test-95 grokking criterion.")
    lines.append("")

    if not lead_times.empty:
        rel = lead_times[lead_times["threshold_type"] == "relative_0.5"].dropna(subset=["lead_time_steps"])
        if not rel.empty:
            agg = (
                rel.groupby("metric_name")["lead_time_steps"]
                .agg(median="median", count="count", fraction_positive=lambda x: float((x > 0).mean()))
                .sort_values("median", ascending=False)
            )
            lines.append("## Relative 50% Lead Times")
            lines.append("")
            lines.append(_markdown_table(agg))
            lines.append("")
            align = agg[agg.index.str.startswith("align_K")]
            baselines = agg[agg.index.isin(["log_weight_norm", "train_loss", "ntk_drift_correct_logit"])]
            if not align.empty:
                best_align = align["median"].idxmax()
                lines.append(f"Best Fourier-alignment median lead time: `{best_align}` = {align.loc[best_align, 'median']:.1f} steps.")
                if not baselines.empty:
                    beaten = int((align.loc[best_align, "median"] > baselines["median"]).sum())
                    lines.append(f"This exceeds {beaten} of {len(baselines)} baseline median lead times.")
    lines.append("")
    lines.append("## Output Files")
    lines.extend(
        [
            "- `metrics_all.csv`",
            "- `agop_spectra_all.csv`",
            "- `lead_times.csv`",
            "- per-run plot folders named by `run_id`",
            "- aggregate `lead_time_bar.*`, `lead_time_table_heatmap.*`, and `final_summary.*`",
        ]
    )
    lines.append("")
    lines.append("Claims should remain cautious: positive lead time is evidence for the AGOP-Fourier hypothesis only when delayed generalization is present.")
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(results_dir: str | Path) -> None:
    results_dir = ensure_dir(results_dir)
    metrics = _read_many(results_dir, "metrics.csv")
    spectra = _read_many(results_dir, "agop_spectra.csv")
    metrics.to_csv(results_dir / "metrics_all.csv", index=False)
    if spectra.empty:
        spectra = pd.DataFrame(columns=["run_id", "seed", "step", "eig_idx", "eigenvalue"])
    spectra.to_csv(results_dir / "agop_spectra_all.csv", index=False)
    lead_times = compute_lead_times(metrics)
    lead_times.to_csv(results_dir / "lead_times.csv", index=False)
    make_all_plots(metrics, spectra, lead_times, results_dir / "plots")
    write_summary(results_dir, metrics, lead_times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    analyze(args.results_dir)


if __name__ == "__main__":
    main()
