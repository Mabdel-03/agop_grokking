#!/usr/bin/env python
"""Select the best fast-track result and write a 3-seed ASAP main config."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asap_root", default="results/asap")
    parser.add_argument("--out_config", default="results/asap_best.yaml")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    rows = []
    for metrics_path in Path(args.asap_root).glob("*/metrics_all.csv"):
        df = pd.read_csv(metrics_path)
        if df.empty:
            continue
        final = df.sort_values("step").iloc[-1]
        rows.append(
            {
                "results_dir": str(metrics_path.parent),
                "max_train_acc": float(df["train_acc"].max()),
                "max_test_acc": float(df["test_acc"].max()),
                "final_test_acc": float(final["test_acc"]),
                "final_train_acc": float(final["train_acc"]),
                "metrics_path": str(metrics_path),
            }
        )
    if not rows:
        raise RuntimeError("No ASAP metrics_all.csv files found")
    summary = pd.DataFrame(rows)
    summary["score"] = summary["max_test_acc"] * 10 + summary["max_train_acc"]
    summary = summary.sort_values("score", ascending=False)
    summary.to_csv(Path(args.asap_root) / "asap_summary.csv", index=False)
    best_dir = Path(summary.iloc[0]["results_dir"])
    cfg_path = best_dir / "run_seed0" / "config_resolved.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["results_dir"] = "results/asap_main"
    cfg["seeds"] = args.seeds
    cfg["checkpoint_preset"] = "main"
    cfg["compute_agop"] = True
    cfg["compute_ntk"] = True
    cfg["save_final_model"] = False
    for volatile in ["run_id", "run_dir", "checkpoint_steps", "device", "git_hash"]:
        cfg.pop(volatile, None)
    out = Path(args.out_config)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(summary.head(8).to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
