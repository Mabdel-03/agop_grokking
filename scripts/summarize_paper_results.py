#!/usr/bin/env python
"""Write a compact cross-suite paper summary from generated result directories."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def markdown_table(df: pd.DataFrame) -> str:
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


def collect_csv(root: Path, name: str) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob(f"**/{name}")):
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if not df.empty:
            df["source_dir"] = str(path.parent)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--out", default="results/paper_summary.md")
    args = parser.parse_args()

    root = Path(args.results_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lead = collect_csv(root, "lead_times.csv")
    metrics = collect_csv(root, "metrics_all.csv")
    lines = ["# Paper Experiment Summary", ""]
    lines.append(f"- Metrics files found: {metrics['source_dir'].nunique() if not metrics.empty else 0}")
    lines.append(f"- Lead-time files found: {lead['source_dir'].nunique() if not lead.empty else 0}")
    lines.append("")
    if not metrics.empty:
        runs = metrics.groupby(["source_dir", "run_id", "seed"], dropna=False)
        lines.append(f"- Total run/seed trajectories: {len(runs)}")
        final = runs.tail(1)
        lines.append(f"- Median final test accuracy: {final['test_acc'].median():.4f}")
        lines.append(f"- Max final test accuracy: {final['test_acc'].max():.4f}")
        lines.append("")
    if not lead.empty and "lead_time_steps" in lead:
        rel = lead[lead["threshold_type"] == "relative_0.5"].dropna(subset=["lead_time_steps"])
        if not rel.empty:
            agg = (
                rel.groupby("metric_name")["lead_time_steps"]
                .agg(median="median", count="count", fraction_positive=lambda x: float((x > 0).mean()))
                .sort_values("median", ascending=False)
            )
            lines.append("## Relative 50% Lead-Time Summary")
            lines.append("")
            lines.append(markdown_table(agg))
            lines.append("")
    lines.append("See each result directory's `summary.md` and `plots/` folder for claim-level interpretation.")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not lead.empty:
        lead.to_csv(root / "paper_lead_times_all.csv", index=False)
    if not metrics.empty:
        metrics.to_csv(root / "paper_metrics_all.csv", index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
