#!/usr/bin/env python
"""Build paper-ready LaTeX table snippets from aim-coverage results.

Output goes to ``--paper-dir`` (defaults to ``<repo>/paper_out``) in two
subdirectories: ``tables/`` (the three aim tables) and ``figures/`` (per-run
plots copied from the result directories).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_PAPER_DIR = REPO_ROOT / "paper_out"


def _median_rel50(lead_path: Path, metrics: list[str]) -> dict[str, float]:
    """Report the median first-crossing step (t_metric) at the relative-50% threshold.

    Falls back from lead_time_steps to t_metric when grokking did not occur and
    t_grok is missing. t_metric is the absolute step at which the metric first
    crosses its relative-0.5 threshold against its end-of-run value.
    """
    if not lead_path.exists():
        return {m: float("nan") for m in metrics}
    df = pd.read_csv(lead_path)
    rel = df[df["threshold_type"] == "relative_0.5"]
    out = {}
    for m in metrics:
        sub = rel[rel["metric_name"] == m].dropna(subset=["t_metric"])
        out[m] = float(sub["t_metric"].median()) if not sub.empty else float("nan")
    return out


def build_aim1_table(results_dir: Path) -> str:
    metrics = ["align_K2", "b_drift_linear1", "b_drift_linear2", "g_drift_linear1", "g_drift_linear2", "ntk_drift_correct_logit", "log_weight_norm", "train_loss"]
    leads = _median_rel50(results_dir / "lead_times.csv", metrics)
    label = {
        "align_K2": "AGOP-Fourier alignment ($K=2$)",
        "b_drift_linear1": "Block backward-Gram $B_{\\mathrm{linear1}}$",
        "b_drift_linear2": "Block backward-Gram $B_{\\mathrm{linear2}}$",
        "g_drift_linear1": "Block forward-Gram $G_{\\mathrm{linear1}}$",
        "g_drift_linear2": "Block forward-Gram $G_{\\mathrm{linear2}}$",
        "ntk_drift_correct_logit": "Full correct-logit NTK drift",
        "log_weight_norm": "Log weight norm",
        "train_loss": "Train loss",
    }
    rows = ["\\begin{tabular}{lc}", "\\toprule", "Metric & First-crossing step \\\\", "\\midrule"]
    for m in metrics:
        v = leads[m]
        rows.append(f"{label[m]} & {v:,.0f} \\\\" if pd.notna(v) else f"{label[m]} & -- \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(rows)


def build_aim2_table(results_dir: Path) -> str:
    metrics = ["vcr_agop_r4", "vcr_agop_r8", "vcr_pca_r4", "vcr_pca_r8", "vcr_random_r4", "vcr_random_r8", "vcr_full"]
    leads = _median_rel50(results_dir / "lead_times.csv", metrics)
    label = {
        "vcr_agop_r4": "$\\mathrm{VCR}_{\\mathrm{AGOP}, r=4}$",
        "vcr_agop_r8": "$\\mathrm{VCR}_{\\mathrm{AGOP}, r=8}$",
        "vcr_pca_r4": "$\\mathrm{VCR}_{\\mathrm{PCA}, r=4}$",
        "vcr_pca_r8": "$\\mathrm{VCR}_{\\mathrm{PCA}, r=8}$",
        "vcr_random_r4": "$\\mathrm{VCR}_{\\mathrm{random}, r=4}$",
        "vcr_random_r8": "$\\mathrm{VCR}_{\\mathrm{random}, r=8}$",
        "vcr_full": "Full-space VCR",
    }
    rows = ["\\begin{tabular}{lc}", "\\toprule", "Projection & First-crossing step \\\\", "\\midrule"]
    for m in metrics:
        v = leads[m]
        rows.append(f"{label[m]} & {v:,.0f} \\\\" if pd.notna(v) else f"{label[m]} & -- \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(rows)


def build_aim3_table(mod_mult_dir: Path, rfm_dir: Path) -> str:
    metrics = ["align_K2", "align_K4"]
    mm_leads = _median_rel50(mod_mult_dir / "lead_times.csv", metrics)
    mm_summary_path = mod_mult_dir / "run_summaries.csv"
    mm_acc = float("nan")
    if mm_summary_path.exists():
        df = pd.read_csv(mm_summary_path)
        mm_acc = float(df["max_test_acc"].max())
    rfm_summary_path = rfm_dir / "run_summaries.csv"
    rfm_lines = []
    if rfm_summary_path.exists():
        rfm_df = pd.read_csv(rfm_summary_path)
        for _, r in rfm_df.iterrows():
            task_label = "modular addition" if r['task'] == 'mod_add' else "modular multiplication"
            rfm_lines.append(f"RFM, {task_label} & {float(r['max_test_acc']) * 100:.1f}\\% & {int(r['iters'])} iters \\\\")
    else:
        rfm_lines.append("RFM (Mahalanobis Gaussian) & not yet generalizing & see next experiments \\\\")
    rows = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Experiment & Quantity & Value \\\\",
        "\\midrule",
        f"Modular multiplication (MLP) & max test accuracy & {mm_acc * 100:.1f}\\% \\\\" if pd.notna(mm_acc) else "Modular multiplication (MLP) & max test accuracy & -- \\\\",
    ]
    rows.append(f"Modular multiplication (MLP) & align $K=2$ first-crossing step & {mm_leads['align_K2']:,.0f} \\\\" if pd.notna(mm_leads['align_K2']) else "Modular multiplication (MLP) & align $K=2$ first-crossing step & -- \\\\")
    if "align_K4" in mm_leads and pd.notna(mm_leads["align_K4"]):
        rows.append(f"Modular multiplication (MLP) & align $K=4$ first-crossing step & {mm_leads['align_K4']:,.0f} \\\\")
    rows.append("\\midrule")
    for line in rfm_lines:
        rows.append(line)
    rows += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(rows)


def _copy_first(matches: list[Path], dest: Path) -> None:
    for p in matches:
        if p.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dest)
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
                        help="Directory containing aim1_block_ntk/, aim2_vcr/, etc.")
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR,
                        help="Output directory; creates tables/ and figures/ inside it.")
    args = parser.parse_args()
    root = args.results_root.resolve()
    paper_dir = args.paper_dir.resolve()
    table_dir = paper_dir / "tables"
    fig_dir = paper_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    aim1 = root / "aim1_block_ntk"
    aim2 = root / "aim2_vcr"
    aim3_mm = root / "aim3_mod_mult"
    aim3_rfm = root / "aim3_rfm"

    (table_dir / "aim1_block_ntk.tex").write_text(build_aim1_table(aim1), encoding="utf-8")
    (table_dir / "aim2_vcr.tex").write_text(build_aim2_table(aim2), encoding="utf-8")
    (table_dir / "aim3_intervention.tex").write_text(build_aim3_table(aim3_mm, aim3_rfm), encoding="utf-8")

    for src_glob, dst_name in [
        ((aim1 / "plots").glob("run_seed0*/agop_alignment_seed0.pdf"), "aim1_alignment_seed0.pdf"),
        ((aim2 / "plots").glob("run_seed0*/normalized_metrics_seed0.pdf"), "aim2_normalized_metrics_seed0.pdf"),
        ((aim2 / "plots").glob("run_seed0*/accuracy_loss_seed0.pdf"), "aim2_accuracy_loss_seed0.pdf"),
        ((aim3_mm / "plots").glob("run_seed0*/agop_alignment_seed0.pdf"), "aim3_mod_mult_alignment_seed0.pdf"),
        ((aim3_mm / "plots").glob("run_seed0*/accuracy_loss_seed0.pdf"), "aim3_mod_mult_accuracy_seed0.pdf"),
    ]:
        _copy_first(list(src_glob), fig_dir / dst_name)

    print(f"Wrote tables to {table_dir}/ and figures to {fig_dir}/")


if __name__ == "__main__":
    main()
