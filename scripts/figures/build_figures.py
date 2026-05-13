"""Generate Figure 2 panels (Aim 1 block-NTK, Aim 2 VCR trajectories) from result CSVs.

Reads metrics.csv from results/aim1_block_ntk/run_seed0/ and results/aim2_vcr/run_seed0/,
writes aim1_block_ntk_traj.pdf and aim2_vcr_traj.pdf to ``--out-dir``.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_OUT_DIR = REPO_ROOT / "paper_out" / "figures"


def _relative(series: pd.Series) -> pd.Series:
    """Min-max normalize so the trajectory lives in [0, 1] regardless of sign."""
    vals = series.dropna()
    lo, hi = float(vals.min()), float(vals.max())
    denom = hi - lo
    if abs(denom) < 1e-12:
        return series * 0.0
    return (series - lo) / denom


def _load_metrics(results_root: Path, aim_subdir: str) -> pd.DataFrame:
    path = results_root / aim_subdir / "run_seed0/metrics.csv"
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. metrics.csv is gitignored; rerun the aim-coverage Slurm job "
            "(scripts/aim_coverage/slurm_run_aims.sbatch) to regenerate it."
        )
    return pd.read_csv(path)


def make_aim1(results_root: Path, out_dir: Path, t_grok: int = 20000) -> None:
    df = _load_metrics(results_root, "aim1_block_ntk")
    df = df.sort_values("step").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for col, label, color, ls in [
        ("b_drift_linear2", r"$B_{\mathrm{linear2}}$ drift", "tab:blue", "-"),
        ("b_drift_linear1", r"$B_{\mathrm{linear1}}$ drift", "tab:orange", "-"),
        ("ntk_drift_correct_logit", "full NTK drift", "tab:green", "-"),
        ("align_K1", r"AGOP-Fourier $K{=}1$", "tab:red", ":"),
        ("align_K2", r"AGOP-Fourier $K{=}2$", "tab:red", "-"),
        ("align_K4", r"AGOP-Fourier $K{=}4$", "tab:red", "--"),
        ("align_K8", r"AGOP-Fourier $K{=}8$", "tab:red", "-."),
    ]:
        if col in df.columns:
            s = _relative(df[col])
            ax.plot(df["step"], s, label=label, color=color, linewidth=1.4, linestyle=ls)
    ax.axvline(t_grok, color="k", linestyle="--", linewidth=1, label=r"$t_{\mathrm{grok}}$")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlim(0, df["step"].max())
    ax.set_xlabel("step")
    ax.set_ylabel("relative progress to end-of-run")
    ax.set_title("Aim 1: block-NTK vs AGOP-Fourier alignment")
    ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "aim1_block_ntk_traj.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir / 'aim1_block_ntk_traj.pdf'}")


def make_aim2(results_root: Path, out_dir: Path, t_grok: int = 20000) -> None:
    df = _load_metrics(results_root, "aim2_vcr")
    df = df.sort_values("step").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for col, label, color in [
        ("vcr_agop_r8", r"AGOP-projected $r{=}8$", "tab:red"),
        ("vcr_pca_r8", r"PCA-projected $r{=}8$", "tab:purple"),
        ("vcr_random_r8", r"random-projected $r{=}8$", "tab:gray"),
        ("vcr_full", "full-space VCR", "tab:brown"),
    ]:
        if col in df.columns:
            ax.plot(df["step"], df[col].clip(lower=1e-3), label=label, color=color, linewidth=1.5)
    ax.axvline(t_grok, color="k", linestyle="--", linewidth=1, label=r"$t_{\mathrm{grok}}$")
    ax.set_xscale("symlog", linthresh=100)
    ax.set_yscale("log")
    ax.set_xlim(0, df["step"].max())
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\mathrm{VCR} = \mathrm{tr}(Q^\top\Sigma_W Q) / \mathrm{tr}(Q^\top\Sigma_B Q)$")
    ax.set_title("Aim 2: projected variation collapse")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "aim2_vcr_traj.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir / 'aim2_vcr_traj.pdf'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--t-grok-addition", type=int, default=20000,
                        help="t_grok for modular addition (Aims 1 and 2).")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    make_aim1(args.results_root, args.out_dir, t_grok=args.t_grok_addition)
    make_aim2(args.results_root, args.out_dir, t_grok=args.t_grok_addition)


if __name__ == "__main__":
    main()
