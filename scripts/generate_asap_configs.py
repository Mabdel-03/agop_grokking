#!/usr/bin/env python
"""Generate a small fast-track config set for same-day AGOP results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


def tag(value: object) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def save_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="results/asap_manifests")
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "mode": "standard",
        "p": 31,
        "train_fraction": 0.5,
        "seeds": args.seeds,
        "model_type": "quadratic",
        "hidden_width": 512,
        "lr": 0.003,
        "weight_decay": 3.0,
        "init_scale": 0.7,
        "max_steps": args.max_steps,
        "batch_size": "full",
        "checkpoint_preset": "pilot_metrics",
        "compute_agop": True,
        "compute_ntk": True,
        "save_final_model": False,
        "agop_batch_size": 128,
        "agop_probe_size": "full",
        "agop_exact_p_max": 47,
        "ntk_probe_size": 64,
        "fourier_K": [1, 2, 4, 8, 16],
        "random_alignment": True,
    }
    configs = []
    for lr in [0.001, 0.003]:
        for weight_decay in [1.0, 3.0, 10.0]:
            for init_scale in [0.7, 1.0]:
                cfg = dict(base)
                cfg.update({"lr": lr, "weight_decay": weight_decay, "init_scale": init_scale})
                name = f"p31_tf0p5_w512_lr{tag(lr)}_wd{tag(weight_decay)}_init{tag(init_scale)}"
                cfg["results_dir"] = f"results/asap/{name}"
                path = out_dir / f"{name}.yaml"
                save_yaml(cfg, path)
                configs.append({"name": name, "config_path": str(path), "results_dir": cfg["results_dir"]})

    relu = dict(base)
    relu.update({"model_type": "relu", "lr": 0.003, "weight_decay": 3.0, "init_scale": 0.7})
    relu_name = "relu_p31_tf0p5_w512_lr0p003_wd3p0_init0p7"
    relu["results_dir"] = f"results/asap/{relu_name}"
    relu_path = out_dir / f"{relu_name}.yaml"
    save_yaml(relu, relu_path)
    configs.append({"name": relu_name, "config_path": str(relu_path), "results_dir": relu["results_dir"]})

    manifest = out_dir / "manifest.csv"
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "config_path", "results_dir"])
        writer.writeheader()
        writer.writerows(configs)
    print(f"Wrote {len(configs)} ASAP configs to {out_dir}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
