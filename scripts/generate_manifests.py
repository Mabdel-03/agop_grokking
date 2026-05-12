#!/usr/bin/env python
"""Generate paper-suite configs from the selected pilot configuration."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def normalized_tag(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def paper_config(base: dict[str, Any], results_dir: str, seeds: list[int], checkpoint_preset: str) -> dict[str, Any]:
    cfg = dict(base)
    cfg.update(
        {
            "mode": "standard",
            "results_dir": results_dir,
            "seeds": seeds,
            "checkpoint_preset": checkpoint_preset,
            "compute_agop": True,
            "compute_ntk": True,
            "save_final_model": True,
            "random_alignment": True,
        }
    )
    return cfg


def add_entry(entries: list[dict[str, str]], name: str, suite: str, cfg: dict[str, Any], out_dir: Path) -> None:
    config_path = out_dir / f"{name}.yaml"
    save_yaml(cfg, config_path)
    entries.append(
        {
            "name": name,
            "suite": suite,
            "config_path": str(config_path),
            "results_dir": str(cfg["results_dir"]),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_config", default="results/pilot/best_config.yaml")
    parser.add_argument("--out_dir", default="results/manifests")
    parser.add_argument("--main_seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--ablation_seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--sensitivity_seeds", nargs="*", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    best_path = Path(args.best_config)
    if not best_path.exists():
        raise FileNotFoundError(f"Best config not found: {best_path}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = load_yaml(best_path)
    best.setdefault("agop_probe_size", 1024)
    best.setdefault("agop_exact_p_max", 47)
    best.setdefault("ntk_probe_size", 128)
    best.setdefault("fourier_K", [1, 2, 4, 8, 16])

    entries: list[dict[str, str]] = []

    main_cfg = paper_config(best, "results/main", args.main_seeds, "main")
    add_entry(entries, "main_best", "main", main_cfg, out_dir)

    for p in [31, 47, 59]:
        cfg = paper_config(best, f"results/sensitivity/p{p}", args.sensitivity_seeds, "pilot_metrics")
        cfg["p"] = p
        if p >= 59:
            cfg["agop_probe_size"] = 1024
        cfg["save_final_model"] = False
        add_entry(entries, f"sensitivity_p{p}", "sensitivity_p", cfg, out_dir)

    for train_fraction in [0.3, 0.4, 0.5]:
        tag = normalized_tag(train_fraction)
        cfg = paper_config(best, f"results/sensitivity/train_fraction_{tag}", args.sensitivity_seeds, "pilot_metrics")
        cfg["train_fraction"] = train_fraction
        cfg["save_final_model"] = False
        add_entry(entries, f"sensitivity_train_fraction_{tag}", "sensitivity_train_fraction", cfg, out_dir)

    relu_cfg = paper_config(best, "results/ablations/relu", args.ablation_seeds, "pilot_metrics")
    relu_cfg["model_type"] = "relu"
    relu_cfg["save_final_model"] = False
    add_entry(entries, "ablation_relu", "ablation", relu_cfg, out_dir)

    frozen_cfg = paper_config(best, "results/ablations/frozen_quadratic", args.ablation_seeds, "pilot_metrics")
    frozen_cfg["model_type"] = "quadratic"
    frozen_cfg["freeze_first_layer"] = True
    frozen_cfg["save_final_model"] = False
    add_entry(entries, "ablation_frozen_quadratic", "ablation", frozen_cfg, out_dir)

    wd0_cfg = paper_config(best, "results/ablations/weight_decay_0", args.ablation_seeds, "pilot_metrics")
    wd0_cfg["weight_decay"] = 0.0
    wd0_cfg["save_final_model"] = False
    add_entry(entries, "ablation_weight_decay_0", "ablation", wd0_cfg, out_dir)

    best_wd = best.get("weight_decay", "best")
    wdbest_cfg = paper_config(best, f"results/ablations/weight_decay_{normalized_tag(best_wd)}", args.ablation_seeds, "pilot_metrics")
    wdbest_cfg["save_final_model"] = False
    add_entry(entries, "ablation_weight_decay_best", "ablation", wdbest_cfg, out_dir)

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "suite", "config_path", "results_dir"])
        writer.writeheader()
        writer.writerows(entries)
    array_manifest_path = out_dir / "array_manifest.csv"
    with open(array_manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "suite", "config_path", "results_dir"])
        writer.writeheader()
        writer.writerows([entry for entry in entries if entry["suite"] != "main"])

    print(f"Wrote {len(entries)} configs to {out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Array manifest: {array_manifest_path}")


if __name__ == "__main__":
    main()
