#!/usr/bin/env python
"""Run one ASAP manifest entry and analyze it immediately."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/asap_manifests/manifest.csv")
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    with open(args.manifest, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.index < 0 or args.index >= len(rows):
        raise IndexError(f"Index {args.index} out of range for {len(rows)} ASAP configs")
    row = rows[args.index]
    print(f"Running ASAP {args.index}: {row['name']}")
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
    subprocess.run([sys.executable, "analyze_results.py", "--results_dir", row["results_dir"]], check=True)


if __name__ == "__main__":
    main()
