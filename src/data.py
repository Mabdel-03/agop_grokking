"""Modular arithmetic datasets with fixed one-hot input coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ModAddDataset:
    X_train: torch.FloatTensor
    y_train: torch.LongTensor
    X_test: torch.FloatTensor
    y_test: torch.LongTensor
    X_full: torch.FloatTensor
    y_full: torch.LongTensor
    meta: dict[str, Any]


def make_mod_add_dataset(
    p: int,
    train_fraction: float,
    seed: int,
) -> tuple[
    torch.FloatTensor,
    torch.LongTensor,
    torch.FloatTensor,
    torch.LongTensor,
    torch.FloatTensor,
    torch.LongTensor,
    dict[str, Any],
]:
    """Create ordered-pair modular addition with x=[onehot(a), onehot(b)]."""
    if p <= 1:
        raise ValueError(f"p must be > 1, got {p}")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0,1), got {train_fraction}")

    pairs = np.array([(a, b) for a in range(p) for b in range(p)], dtype=np.int64)
    y_np = ((pairs[:, 0] + pairs[:, 1]) % p).astype(np.int64)

    X_np = np.zeros((p * p, 2 * p), dtype=np.float32)
    rows = np.arange(p * p)
    X_np[rows, pairs[:, 0]] = 1.0
    X_np[rows, p + pairs[:, 1]] = 1.0

    rng = np.random.default_rng(seed)
    perm = rng.permutation(p * p)
    n_train = int(round(train_fraction * p * p))
    n_train = min(max(n_train, 1), p * p - 1)
    train_idx = np.sort(perm[:n_train])
    test_idx = np.sort(perm[n_train:])

    X_full = torch.from_numpy(X_np).float()
    y_full = torch.from_numpy(y_np).long()
    X_train = X_full[torch.from_numpy(train_idx).long()]
    y_train = y_full[torch.from_numpy(train_idx).long()]
    X_test = X_full[torch.from_numpy(test_idx).long()]
    y_test = y_full[torch.from_numpy(test_idx).long()]

    meta = {
        "p": int(p),
        "train_fraction": float(train_fraction),
        "seed": int(seed),
        "train_indices": train_idx,
        "test_indices": test_idx,
        "pairs": pairs,
    }
    return X_train, y_train, X_test, y_test, X_full, y_full, meta


def validate_mod_add_dataset(
    X_full: torch.Tensor,
    y_full: torch.Tensor,
    meta: dict[str, Any],
) -> list[str]:
    """Return human-readable dataset check results; raise on hard failures."""
    p = int(meta["p"])
    pairs = np.asarray(meta["pairs"])
    train_idx = set(np.asarray(meta["train_indices"]).tolist())
    test_idx = set(np.asarray(meta["test_indices"]).tolist())
    messages: list[str] = []

    assert X_full.shape == (p * p, 2 * p), X_full.shape
    messages.append(f"X_full shape OK: {tuple(X_full.shape)}")
    row_sums = X_full.sum(dim=1)
    assert torch.allclose(row_sums, torch.full_like(row_sums, 2.0)), row_sums
    messages.append("Each row has exactly two active one-hot coordinates")
    expected_y = torch.tensor((pairs[:, 0] + pairs[:, 1]) % p, dtype=torch.long)
    assert torch.equal(y_full.cpu(), expected_y), "labels do not match modular addition"
    messages.append("Labels match (a+b) mod p")
    assert train_idx.isdisjoint(test_idx), "train/test split overlaps"
    messages.append("Train/test split is disjoint")
    return messages
