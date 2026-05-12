#!/usr/bin/env python
"""Command-line entrypoint for AGOP grokking experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.train import run_pilot_search, run_standard_experiments
from src.utils import load_yaml, merge_overrides, save_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--results_dir", default=None, help="Override output directory")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Override seed list")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = merge_overrides(
        cfg,
        {
            "results_dir": args.results_dir,
            "max_steps": args.max_steps,
            "device": args.device,
        },
    )
    if args.seeds is not None and len(args.seeds) > 0:
        cfg["seeds"] = args.seeds

    results_dir = Path(cfg.get("results_dir", args.results_dir or "results/main"))
    results_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, results_dir / "launch_config.yaml")

    mode = cfg.get("mode", "standard").lower()
    if mode == "pilot":
        run_pilot_search(cfg, device=args.device, progress=not args.no_progress)
    elif mode == "standard":
        run_standard_experiments(
            cfg,
            results_dir=cfg.get("results_dir", args.results_dir),
            seeds=cfg.get("seeds", [0]),
            device=args.device,
            progress=not args.no_progress,
        )
    else:
        raise ValueError(f"Unknown mode={mode!r}; expected standard or pilot")


if __name__ == "__main__":
    main()
