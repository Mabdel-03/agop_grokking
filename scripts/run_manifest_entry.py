#!/usr/bin/env python
"""Run one generated manifest entry and analyze its isolated result directory."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/manifests/manifest.csv")
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    with open(manifest, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.index < 0 or args.index >= len(rows):
        raise IndexError(f"Manifest index {args.index} out of range for {len(rows)} entries")
    row = rows[args.index]
    print(f"Running manifest row {args.index}: {row['name']} ({row['suite']})")
    subprocess.run(
        [
            sys.executable,
            "run_experiment.py",
            "--config",
            row["config_path"],
            "--results_dir",
            row["results_dir"],
            "--device",
            args.device,
            "--no_progress",
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, "analyze_results.py", "--results_dir", row["results_dir"]],
        check=True,
    )


if __name__ == "__main__":
    main()
