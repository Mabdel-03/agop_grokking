"""Recursive Feature Machine for modular arithmetic (Aim 3 intervention).

We implement the canonical iteration of Radhakrishnan et al. and Mallinar
et al.: at each iteration t, solve ridge regression with a Mahalanobis
Gaussian kernel K_M(x, x') = exp(-(x-x')^T M (x-x') / (2 L^2)), compute the
AGOP of the resulting predictor, and set M_{t+1} = AGOP.

This is the matched intervention promised by Aim 3, run on the same data
splits as the neural network so AGOP-Fourier alignment trajectories are
directly comparable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .data import make_mod_add_dataset, make_mod_mult_dataset
from .fourier import (
    alignment_from_agop,
    make_fourier_basis,
    make_random_basis,
    subspace_alignment_from_eigenvectors,
    valid_fourier_K,
)
from .metrics import effective_rank


def _mahalanobis_kernel(X: torch.Tensor, Y: torch.Tensor, M: torch.Tensor, L: float) -> torch.Tensor:
    Xm = X @ M
    Ym = Y @ M
    XX = (X * Xm).sum(dim=1, keepdim=True)
    YY = (Y * Ym).sum(dim=1, keepdim=True)
    XY = X @ Ym.T
    dist2 = XX + YY.T - 2.0 * XY
    dist2 = dist2.clamp_min(0.0)
    return torch.exp(-dist2 / (2.0 * (L ** 2)))


def _rfm_predict(X_train: torch.Tensor, alpha: torch.Tensor, X_query: torch.Tensor, M: torch.Tensor, L: float) -> torch.Tensor:
    K = _mahalanobis_kernel(X_query, X_train, M, L)
    return K @ alpha


def _compute_agop_rfm(
    X_train: torch.Tensor,
    alpha: torch.Tensor,
    X_probe: torch.Tensor,
    M: torch.Tensor,
    L: float,
) -> torch.Tensor:
    """A = (1/n) sum_i J(x_i)^T J(x_i) for f(x) = K_M(X_train, x) alpha.

    Closed form: J(x)_{c,j} = sum_i K_M(x_i, x) (-(x - x_i)^T M)_j alpha_{i,c} / L^2.
    """
    n_probe = X_probe.shape[0]
    n_train = X_train.shape[0]
    d = X_probe.shape[1]
    A = torch.zeros(d, d, dtype=torch.float64)
    for j in range(n_probe):
        x = X_probe[j].unsqueeze(0)
        K_row = _mahalanobis_kernel(x, X_train, M, L).squeeze(0)
        diff = (x - X_train) @ M
        coeffs = (K_row.unsqueeze(1) * alpha) / (L ** 2)
        J = (-diff.T @ coeffs).T
        A += (J.T @ J).to(torch.float64)
    A /= max(n_probe, 1)
    A = 0.5 * (A + A.T)
    return A


def run_rfm(
    p: int,
    task: str,
    train_fraction: float,
    seed: int,
    iters: int,
    L: float,
    ridge: float,
    fourier_K: list[int],
    out_dir: Path,
) -> dict[str, Any]:
    if task == "mod_mult":
        X_train, y_train, X_test, y_test, X_full, y_full, meta = make_mod_mult_dataset(p, train_fraction, seed)
    else:
        X_train, y_train, X_test, y_test, X_full, y_full, meta = make_mod_add_dataset(p, train_fraction, seed)

    m = int(meta.get("m", p))
    X_tr = X_train.to(torch.float64)
    X_te = X_test.to(torch.float64)
    n_classes = int(m if task == "mod_mult" else p)
    Y_tr = torch.zeros(X_tr.shape[0], n_classes, dtype=torch.float64)
    Y_tr[torch.arange(X_tr.shape[0]), y_train.long()] = 1.0
    Y_te = y_test.long()

    d = X_tr.shape[1]
    M = torch.eye(d, dtype=torch.float64)

    valid_K = valid_fourier_K(p, fourier_K, m=m)
    fourier_Q = {K: make_fourier_basis(p, K, m=m)[0] for K in valid_K}
    random_P = {K: make_random_basis(2 * m, fourier_Q[K].shape[1], seed=seed + 1000 + K)[1] for K in valid_K}

    rows: list[dict[str, Any]] = []
    spectra_rows: list[dict[str, Any]] = []

    run_id = f"rfm_seed{seed}_p{p}_{task}_tf{train_fraction}_L{L}_ridge{ridge}".replace(".", "p")
    out_dir.mkdir(parents=True, exist_ok=True)

    for t in range(iters + 1):
        K_train = _mahalanobis_kernel(X_tr, X_tr, M, L)
        alpha = torch.linalg.solve(K_train + ridge * torch.eye(X_tr.shape[0], dtype=torch.float64), Y_tr)

        train_logits = _rfm_predict(X_tr, alpha, X_tr, M, L)
        test_logits = _rfm_predict(X_tr, alpha, X_te, M, L)
        train_acc = float((train_logits.argmax(dim=-1).cpu() == y_train.long()).float().mean().item())
        test_acc = float((test_logits.argmax(dim=-1).cpu() == Y_te).float().mean().item())

        A = _compute_agop_rfm(X_tr, alpha, X_tr, M, L)
        evals_full = torch.linalg.eigvalsh(A)
        evals_full = evals_full[torch.argsort(evals_full, descending=True)]
        eff_rank = effective_rank(evals_full)

        row: dict[str, Any] = {
            "run_id": run_id,
            "seed": int(seed),
            "p": int(p),
            "task": task,
            "train_fraction": float(train_fraction),
            "model_type": "rfm",
            "freeze_first_layer": False,
            "hidden_width": int(X_tr.shape[1]),
            "lr": float(L),
            "weight_decay": float(ridge),
            "init_scale": 1.0,
            "step": int(t),
            "train_loss": float(((train_logits - Y_tr) ** 2).mean().item()),
            "test_loss": float("nan"),
            "train_acc": train_acc,
            "test_acc": test_acc,
            "weight_norm": float(torch.linalg.matrix_norm(M).item()),
            "log_weight_norm": float(torch.log(torch.linalg.matrix_norm(M) + 1e-12).item()),
            "elapsed_seconds": 0.0,
            "agop_trace": float(torch.trace(A).item()),
            "agop_top_eval_1": float(evals_full[0].item()) if len(evals_full) else float("nan"),
            "agop_top_eval_2": float(evals_full[1].item()) if len(evals_full) > 1 else float("nan"),
            "agop_top_eval_5": float(evals_full[4].item()) if len(evals_full) > 4 else float("nan"),
            "agop_effective_rank": float(eff_rank),
            "agop_normalized_effective_rank": float(eff_rank / max(d, 1)),
            "agop_eigengap_1_2": float((evals_full[0] - evals_full[1]).item()) if len(evals_full) > 1 else float("nan"),
            "agop_min_eval": float(evals_full[-1].item()) if len(evals_full) else float("nan"),
            "agop_symmetry_error": 0.0,
            "ntk_drift_correct_logit": float("nan"),
        }
        for K, Q in fourier_Q.items():
            align = alignment_from_agop(A, Q)
            row[f"align_K{K}"] = float(align["alignment"])
            row[f"raw_overlap_K{K}"] = float(align["raw_overlap"])
            s = Q.shape[1]
            U_r = align["eigenvectors"][:, :s]
            rand_align, rand_raw = subspace_alignment_from_eigenvectors(U_r, random_P[K])
            row[f"random_align_K{K}"] = float(rand_align)
            row[f"random_raw_overlap_K{K}"] = float(rand_raw)
        rows.append(row)
        for idx, val in enumerate(evals_full[: min(2 * p, 50)].tolist(), start=1):
            spectra_rows.append({"run_id": run_id, "seed": int(seed), "step": int(t), "eig_idx": idx, "eigenvalue": float(val)})

        scale = torch.trace(A).clamp_min(1e-12)
        M = (A / scale * d).clone()

    df = pd.DataFrame(rows)
    spectra_df = pd.DataFrame(spectra_rows)
    df.to_csv(out_dir / "metrics.csv", index=False)
    spectra_df.to_csv(out_dir / "agop_spectra.csv", index=False)

    summary = {
        "run_id": run_id,
        "seed": int(seed),
        "p": int(p),
        "task": task,
        "iters": int(iters),
        "L": float(L),
        "ridge": float(ridge),
        "max_train_acc": float(df["train_acc"].max()),
        "max_test_acc": float(df["test_acc"].max()),
        "final_train_acc": float(df["train_acc"].iloc[-1]),
        "final_test_acc": float(df["test_acc"].iloc[-1]),
        "grokking_like": bool(df["test_acc"].iloc[-1] >= 0.9),
    }
    return summary
