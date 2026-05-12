"""Non-interactive plotting for AGOP grokking experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import ensure_dir


def _save(fig: plt.Figure, out_base: Path) -> None:
    ensure_dir(out_base.parent)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=200)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def _vline(ax: plt.Axes, x: float | None, label: str, color: str) -> None:
    if x is not None and not np.isnan(x):
        ax.axvline(x, color=color, linestyle="--", linewidth=1.3, label=label)


def plot_accuracy_loss(df: pd.DataFrame, out_dir: Path, seed: int, t_train_fit: int | None, t_grok: int | None) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(df["step"], df["train_acc"], label="train acc")
    axes[0].plot(df["step"], df["test_acc"], label="test acc")
    _vline(axes[0], t_train_fit, "train fit", "tab:gray")
    _vline(axes[0], t_grok, "grok", "tab:red")
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(loc="best")
    axes[0].set_title(f"Accuracy and loss, seed {seed}")

    axes[1].plot(df["step"], df["train_loss"], label="train loss")
    axes[1].plot(df["step"], df["test_loss"], label="test loss")
    _vline(axes[1], t_train_fit, "train fit", "tab:gray")
    _vline(axes[1], t_grok, "grok", "tab:red")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("cross entropy")
    axes[1].legend(loc="best")
    _save(fig, out_dir / f"accuracy_loss_seed{seed}")


def plot_agop_alignment(df: pd.DataFrame, out_dir: Path, seed: int, t_grok: int | None) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(df["step"], df["test_acc"], color="black", linewidth=2, label="test acc")
    ax1.set_ylabel("test accuracy")
    ax1.set_ylim(-0.02, 1.02)
    ax2 = ax1.twinx()
    for col in [c for c in df.columns if c.startswith("align_K")]:
        ax2.plot(df["step"], df[col], label=col)
    ax2.set_ylabel("normalized alignment")
    _vline(ax1, t_grok, "grok", "tab:red")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_xlabel("step")
    ax1.set_title(f"AGOP-Fourier alignment, seed {seed}")
    _save(fig, out_dir / f"agop_alignment_seed{seed}")


def _relative_series(s: pd.Series, decreasing: bool = False) -> pd.Series:
    vals = s.astype(float)
    if decreasing:
        vals = vals.iloc[0] - vals
    else:
        vals = vals - vals.iloc[0]
    denom = vals.iloc[-1]
    if abs(float(denom)) < 1e-12:
        return vals * np.nan
    return vals / denom


def plot_normalized_metrics(df: pd.DataFrame, out_dir: Path, seed: int, t_grok: int | None) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    align_cols = [c for c in df.columns if c.startswith("align_K")]
    if align_cols:
        best_col = max(align_cols, key=lambda c: df[c].max(skipna=True))
        ax.plot(df["step"], _relative_series(df[best_col]), label=best_col)
    if "log_weight_norm" in df:
        ax.plot(df["step"], _relative_series(df["log_weight_norm"]), label="log_weight_norm")
    if "ntk_drift_correct_logit" in df:
        ax.plot(df["step"], _relative_series(df["ntk_drift_correct_logit"].ffill().fillna(0)), label="ntk drift")
    if "train_loss" in df:
        ax.plot(df["step"], _relative_series(df["train_loss"], decreasing=True), label="train loss progress")
    _vline(ax, t_grok, "grok", "tab:red")
    ax.axhline(0.5, color="tab:gray", linestyle=":", linewidth=1)
    ax.axhline(0.8, color="tab:gray", linestyle=":", linewidth=1)
    ax.set_xlabel("step")
    ax.set_ylabel("relative progress")
    ax.set_title(f"Normalized metric progress, seed {seed}")
    ax.legend(loc="best")
    _save(fig, out_dir / f"normalized_metrics_seed{seed}")


def plot_agop_spectrum(spectra: pd.DataFrame, out_dir: Path, seed: int) -> None:
    if spectra.empty:
        return
    spec = spectra[spectra["seed"] == seed]
    if spec.empty:
        return
    top = spec[spec["eig_idx"] <= 20]
    pivot = top.pivot_table(index="eig_idx", columns="step", values="eigenvalue", aggfunc="first")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(np.log10(pivot.clip(lower=1e-12)), ax=ax, cmap="mako", cbar_kws={"label": "log10 eigenvalue"})
    ax.set_title(f"AGOP spectrum, seed {seed}")
    ax.set_xlabel("step")
    ax.set_ylabel("eigenvalue rank")
    _save(fig, out_dir / f"agop_spectrum_seed{seed}")


def plot_random_vs_fourier(df: pd.DataFrame, out_dir: Path, seed: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    any_line = False
    for col in [c for c in df.columns if c.startswith("align_K")]:
        K = col.replace("align_K", "")
        rand = f"random_align_K{K}"
        ax.plot(df["step"], df[col], label=col)
        any_line = True
        if rand in df.columns:
            ax.plot(df["step"], df[rand], linestyle="--", label=rand)
    if not any_line:
        plt.close(fig)
        return
    ax.set_xlabel("step")
    ax.set_ylabel("normalized alignment")
    ax.set_title(f"Fourier vs random alignment, seed {seed}")
    ax.legend(loc="best")
    _save(fig, out_dir / f"random_vs_fourier_alignment_seed{seed}")


def plot_lead_time_bar(lead_times: pd.DataFrame, out_dir: Path) -> None:
    if lead_times.empty:
        return
    rel = lead_times[lead_times["threshold_type"].isin(["relative_0.5", "relative_0.8"])].copy()
    rel = rel.dropna(subset=["lead_time_steps"])
    if rel.empty:
        return
    summary = (
        rel.groupby("metric_name")["lead_time_steps"]
        .agg(median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
        .reset_index()
        .sort_values("median", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    yerr = np.vstack([summary["median"] - summary["q25"], summary["q75"] - summary["median"]])
    ax.bar(summary["metric_name"], summary["median"], yerr=yerr, capsize=4)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("median lead time (steps)")
    ax.set_title("Lead time by metric")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, out_dir / "lead_time_bar")


def plot_lead_time_heatmap(lead_times: pd.DataFrame, out_dir: Path) -> None:
    if lead_times.empty:
        return
    rel = lead_times[lead_times["threshold_type"] == "relative_0.5"].copy()
    rel = rel.dropna(subset=["lead_time_steps"])
    if rel.empty:
        return
    pivot = rel.pivot_table(index="metric_name", columns="seed", values="lead_time_steps", aggfunc="median")
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(pivot))))
    sns.heatmap(pivot, annot=True, fmt=".0f", center=0, cmap="vlag", ax=ax)
    ax.set_title("Lead times by seed, relative 50% threshold")
    _save(fig, out_dir / "lead_time_table_heatmap")


def plot_final_summary(metrics: pd.DataFrame, lead_times: pd.DataFrame, out_dir: Path) -> None:
    if metrics.empty:
        return
    seed = int(metrics["seed"].iloc[0])
    df = metrics[metrics["seed"] == seed].sort_values("step")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(df["step"], df["train_acc"], label="train")
    axes[0, 0].plot(df["step"], df["test_acc"], label="test")
    axes[0, 0].set_title("Accuracy")
    axes[0, 0].legend()
    for col in [c for c in df.columns if c.startswith("align_K")]:
        axes[0, 1].plot(df["step"], df[col], label=col)
    axes[0, 1].set_title("AGOP-Fourier alignment")
    axes[0, 1].legend(fontsize=8)
    align_cols = [c for c in df.columns if c.startswith("align_K")]
    if align_cols:
        best_col = max(align_cols, key=lambda c: df[c].max(skipna=True))
        axes[1, 0].plot(df["step"], _relative_series(df[best_col]), label=best_col)
    if "log_weight_norm" in df:
        axes[1, 0].plot(df["step"], _relative_series(df["log_weight_norm"]), label="weight")
    if "ntk_drift_correct_logit" in df:
        axes[1, 0].plot(df["step"], _relative_series(df["ntk_drift_correct_logit"].ffill().fillna(0)), label="ntk")
    axes[1, 0].set_title("Normalized metrics")
    axes[1, 0].legend(fontsize=8)
    if not lead_times.empty:
        rel = lead_times[lead_times["threshold_type"] == "relative_0.5"].dropna(subset=["lead_time_steps"])
        if not rel.empty:
            med = rel.groupby("metric_name")["lead_time_steps"].median().sort_values(ascending=False)
            axes[1, 1].bar(med.index, med.values)
            axes[1, 1].tick_params(axis="x", rotation=45)
            axes[1, 1].axhline(0, color="black", linewidth=1)
    axes[1, 1].set_title("Lead time")
    _save(fig, out_dir / "final_summary")


def make_all_plots(metrics: pd.DataFrame, spectra: pd.DataFrame, lead_times: pd.DataFrame, out_dir: str | Path) -> None:
    out_dir = ensure_dir(out_dir)
    for (run_id, seed), df in metrics.groupby(["run_id", "seed"], dropna=False):
        df = df.sort_values("step")
        run_out = ensure_dir(out_dir / str(run_id))
        lt = lead_times[(lead_times["run_id"] == run_id) & (lead_times["seed"] == seed)] if not lead_times.empty else pd.DataFrame()
        t_train_fit = int(lt["t_train_fit"].dropna().iloc[0]) if not lt.empty and lt["t_train_fit"].notna().any() else None
        t_grok = int(lt["t_grok"].dropna().iloc[0]) if not lt.empty and lt["t_grok"].notna().any() else None
        seed_int = int(seed)
        plot_accuracy_loss(df, run_out, seed_int, t_train_fit, t_grok)
        plot_agop_alignment(df, run_out, seed_int, t_grok)
        plot_normalized_metrics(df, run_out, seed_int, t_grok)
        plot_random_vs_fourier(df, run_out, seed_int)
        plot_agop_spectrum(spectra, run_out, seed_int)
    plot_lead_time_bar(lead_times, out_dir)
    plot_lead_time_heatmap(lead_times, out_dir)
    plot_final_summary(metrics, lead_times, out_dir)
