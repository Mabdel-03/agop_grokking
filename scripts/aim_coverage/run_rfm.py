#!/usr/bin/env python
"""Driver for the matched RFM intervention (Aim 3)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rfm import run_rfm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--iters", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seeds = args.seeds if args.seeds is not None else list(cfg.get("seeds", [0, 1]))
    results_dir = Path(args.results_dir or cfg.get("results_dir", "results/aim3_rfm"))
    results_dir.mkdir(parents=True, exist_ok=True)
    iters = int(args.iters if args.iters is not None else cfg.get("iters", 20))
    tasks = list(cfg.get("tasks", ["mod_add", "mod_mult"]))

    summaries = []
    for task in tasks:
        for seed in seeds:
            out_dir = results_dir / f"run_seed{seed}_{task}"
            summary = run_rfm(
                p=int(cfg["p"]),
                task=task,
                train_fraction=float(cfg["train_fraction"]),
                seed=int(seed),
                iters=iters,
                L=float(cfg.get("L", 10.0)),
                ridge=float(cfg.get("ridge", 1e-3)),
                fourier_K=list(cfg.get("fourier_K", [1, 2, 4, 8])),
                out_dir=out_dir,
            )
            summary["run_dir"] = str(out_dir)
            summary["t_train_fit"] = None
            summary["t_grok"] = None
            summaries.append(summary)

    pd.DataFrame(summaries).to_csv(results_dir / "run_summaries.csv", index=False)
    metric_paths = sorted(results_dir.glob("**/metrics.csv"))
    spectrum_paths = sorted(results_dir.glob("**/agop_spectra.csv"))
    if metric_paths:
        pd.concat([pd.read_csv(p) for p in metric_paths], ignore_index=True).to_csv(results_dir / "metrics_all.csv", index=False)
    if spectrum_paths:
        pd.concat([pd.read_csv(p) for p in spectrum_paths], ignore_index=True).to_csv(results_dir / "agop_spectra_all.csv", index=False)
    print(f"Wrote {len(summaries)} RFM runs to {results_dir}")


if __name__ == "__main__":
    main()
