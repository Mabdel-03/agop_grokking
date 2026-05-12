"""Shared utilities for reproducible AGOP grokking experiments."""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml


EPS = 1e-12


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_builtin(data), f, sort_keys=False)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(data), f, indent=2, sort_keys=True)


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(device: str | None = None) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == y).float().mean().item())


def iter_batches(X: torch.Tensor, batch_size: int | str | None) -> Iterable[torch.Tensor]:
    if batch_size in (None, "full") or int(batch_size) >= len(X):
        yield X
        return
    bs = int(batch_size)
    for start in range(0, len(X), bs):
        yield X[start : start + bs]


def flatten_config(prefix: str, cfg: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            flat.update(flatten_config(key, v))
        else:
            flat[key] = v
    return flat


def merge_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    for key, value in overrides.items():
        if value is not None:
            out[key] = value
    return out


def file_exists(path: str | Path) -> bool:
    return Path(path).exists()
