"""Fourier subspaces and alignment metrics in fixed modular input coordinates."""

from __future__ import annotations

import math

import torch


def valid_fourier_K(p: int, requested: list[int] | tuple[int, ...], m: int | None = None) -> list[int]:
    mod = int(m if m is not None else p)
    max_k = (mod - 1) // 2
    d = 2 * mod
    return [int(k) for k in requested if int(k) >= 1 and int(k) <= max_k and 4 * int(k) < d]


def make_fourier_basis(p: int, K: int, dtype: torch.dtype = torch.float64, m: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Q and P for the real Fourier subspace S*_K in R^{2m}; m defaults to p."""
    mod = int(m if m is not None else p)
    if K < 1 or K > (mod - 1) // 2:
        raise ValueError(f"K={K} invalid for modulus m={mod}")
    d = 2 * mod
    t = torch.arange(mod, dtype=dtype)
    cols = []
    for k in range(1, K + 1):
        angle = 2.0 * math.pi * k * t / mod
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        for block, vec in ((0, cos), (0, sin), (1, cos), (1, sin)):
            col = torch.zeros(d, dtype=dtype)
            start = block * mod
            col[start : start + mod] = vec
            cols.append(col)
    B = torch.stack(cols, dim=1)
    Q, _ = torch.linalg.qr(B, mode="reduced")
    P = Q @ Q.T
    return Q, P


def make_random_basis(d: int, s: int, seed: int, dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    B = torch.randn(d, s, generator=gen, dtype=dtype)
    Q, _ = torch.linalg.qr(B, mode="reduced")
    return Q, Q @ Q.T


def subspace_alignment_from_eigenvectors(U_r: torch.Tensor, P: torch.Tensor) -> tuple[float, float]:
    """Return normalized alignment and raw overlap for top-r eigenvectors and projector P."""
    U_r = U_r.to(dtype=P.dtype, device=P.device)
    r = U_r.shape[1]
    s = int(round(float(torch.trace(P).item())))
    d = P.shape[0]
    raw = float(torch.linalg.matrix_norm(U_r.T @ P, ord="fro").square().item() / max(r, 1))
    chance = s / d
    normalized = (raw - chance) / max(1.0 - chance, 1e-12)
    return float(normalized), float(raw)


def alignment_from_agop(A: torch.Tensor, Q: torch.Tensor) -> dict[str, float | torch.Tensor]:
    """Eigendiagonalize A and align its top s-dimensional subspace with span(Q)."""
    A64 = A.detach().cpu().to(torch.float64)
    Q64 = Q.detach().cpu().to(torch.float64)
    P = Q64 @ Q64.T
    evals, evecs = torch.linalg.eigh(A64)
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    evecs = evecs[:, order]
    s = Q64.shape[1]
    U_r = evecs[:, :s]
    align, raw = subspace_alignment_from_eigenvectors(U_r, P)
    return {"alignment": align, "raw_overlap": raw, "eigenvalues": evals, "eigenvectors": evecs}


def validate_fourier(p: int, K: int, trials: int = 20, m: int | None = None) -> list[str]:
    mod = int(m if m is not None else p)
    messages: list[str] = []
    Q, P = make_fourier_basis(p, K, m=mod)
    I = torch.eye(Q.shape[1], dtype=Q.dtype)
    assert torch.allclose(Q.T @ Q, I, atol=1e-8), "Fourier Q is not orthonormal"
    messages.append(f"Q.T @ Q close to identity for K={K}, m={mod}")
    assert torch.allclose(P, P.T, atol=1e-8), "Fourier projector is not symmetric"
    assert torch.allclose(P @ P, P, atol=1e-8), "Fourier projector is not idempotent"
    messages.append("P is symmetric and idempotent")
    perfect, _ = subspace_alignment_from_eigenvectors(Q, P)
    assert abs(perfect - 1.0) < 1e-8, f"perfect alignment is {perfect}"
    messages.append("Perfect Fourier alignment is 1")
    d, s = 2 * mod, Q.shape[1]
    vals = []
    for seed in range(trials):
        R, _ = make_random_basis(d, s, seed)
        vals.append(subspace_alignment_from_eigenvectors(R, P)[0])
    mean_random = sum(vals) / len(vals)
    assert abs(mean_random) < 0.35, f"random alignment mean unexpectedly large: {mean_random}"
    messages.append(f"Random-subspace normalized alignment mean near 0: {mean_random:.3f}")
    return messages
