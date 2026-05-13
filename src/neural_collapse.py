"""Held-out projected neural collapse diagnostics (Aim 2).

For the two-layer quadratic / ReLU MLP, the hidden state z(x) is the
post-activation output of `model.linear1`. We compute class-conditional
within- and between-class scatters on held-out examples, then form
projected variation collapse ratios

    VCR_Q(t) = tr(Q^T Sigma_W Q) / (tr(Q^T Sigma_B Q) + eta)

where Q is one of: the top-r AGOP-eigendirection projector lifted to hidden
space, the top-r PCA projector of held-out hidden states, an orthonormal
random subspace of the same dimension, or the identity (full-space VCR).

The AGOP eigenvectors live in input space (d = 2p), but VCR operates in
hidden space (d_h = hidden_width). We lift each input-space direction v
into hidden space via the linear-1 Jacobian: for the quadratic model,
h(x) = (W_1 x + b_1)^2, so d h_k / d x_j = 2 (W_1 x + b_1)_k W_1[k, j].
We approximate the lifted direction by W_1 v, evaluated on the held-out
batch and ortho-normalized.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .fourier import make_random_basis
from .models import QuadraticMLP, ReLUMLP


@torch.no_grad()
def _hidden_states(model: torch.nn.Module, X: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    device = torch.device(device)
    model.eval()
    pre = model.linear1(X.to(device))
    if isinstance(model, QuadraticMLP):
        z = pre.square()
    elif isinstance(model, ReLUMLP):
        z = F.relu(pre)
    else:
        z = pre
    return z.detach().to(torch.float64).cpu()


def _class_scatters(Z: torch.Tensor, y: torch.Tensor, n_classes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Z = Z.to(torch.float64)
    y = y.cpu().long()
    mu = Z.mean(dim=0)
    mu_c = torch.zeros(n_classes, Z.shape[1], dtype=torch.float64)
    counts = torch.zeros(n_classes, dtype=torch.float64)
    for c in range(n_classes):
        mask = y == c
        if int(mask.sum().item()) > 0:
            mu_c[c] = Z[mask].mean(dim=0)
            counts[c] = float(mask.sum().item())
        else:
            mu_c[c] = mu
    diffs_w = Z - mu_c[y]
    Sigma_W = diffs_w.T @ diffs_w / max(Z.shape[0], 1)
    diffs_b = mu_c - mu.unsqueeze(0)
    Sigma_B = diffs_b.T @ diffs_b / max(n_classes, 1)
    return mu_c, Sigma_W, Sigma_B


def _projected_vcr(Sigma_W: torch.Tensor, Sigma_B: torch.Tensor, Q: torch.Tensor, eta: float = 1e-6) -> float:
    Qd = Q.to(Sigma_W.dtype)
    num = float(torch.trace(Qd.T @ Sigma_W @ Qd).item())
    den = float(torch.trace(Qd.T @ Sigma_B @ Qd).item()) + eta
    return num / den


def _lift_input_directions(W1: torch.Tensor, V_in: torch.Tensor) -> torch.Tensor:
    """Lift r input-space directions (d x r) to hidden space (d_h x r) via W1 v, ortho-normalize."""
    lifted = W1.to(V_in.dtype) @ V_in  # d_h x r
    Q, _ = torch.linalg.qr(lifted, mode="reduced")
    return Q


def vcr_diagnostics(
    model: torch.nn.Module,
    X_holdout: torch.Tensor,
    y_holdout: torch.Tensor,
    A_input: torch.Tensor,
    n_classes: int,
    seed: int,
    device: torch.device | str = "cuda",
    rs: tuple[int, ...] = (4, 8),
) -> dict[str, float]:
    """Compute projected and full-space VCR on held-out hidden states."""
    Z = _hidden_states(model, X_holdout, device)
    d_h = Z.shape[1]
    mu_c, Sigma_W, Sigma_B = _class_scatters(Z, y_holdout, n_classes)

    W1 = model.linear1.weight.detach().cpu().to(torch.float64)  # d_h x d_in
    A = A_input.detach().cpu().to(torch.float64)
    A_sym = 0.5 * (A + A.T)
    A_ridge = 1e-8 * float(torch.diagonal(A_sym).abs().mean().clamp_min(1e-12).item())
    A_reg = A_sym + A_ridge * torch.eye(A_sym.shape[0], dtype=A_sym.dtype)
    try:
        evals, evecs = torch.linalg.eigh(A_reg)
    except torch._C._LinAlgError:
        U, S, _ = torch.linalg.svd(A_reg)
        evals = S
        evecs = U
    order = torch.argsort(evals, descending=True)
    evecs = evecs[:, order]

    Sigma_total = Sigma_W + Sigma_B
    Sigma_total_sym = 0.5 * (Sigma_total + Sigma_total.T)
    ridge = 1e-6 * float(torch.diagonal(Sigma_total_sym).abs().mean().clamp_min(1e-12).item())
    Sigma_total_reg = Sigma_total_sym + ridge * torch.eye(d_h, dtype=Sigma_total_sym.dtype)
    try:
        pca_evals, pca_evecs = torch.linalg.eigh(Sigma_total_reg)
    except torch._C._LinAlgError:
        U, S, _ = torch.linalg.svd(Sigma_total_reg)
        pca_evals = S
        pca_evecs = U
    pca_order = torch.argsort(pca_evals, descending=True)
    pca_evecs = pca_evecs[:, pca_order]

    out: dict[str, float] = {}
    out["vcr_full"] = _projected_vcr(Sigma_W, Sigma_B, torch.eye(d_h, dtype=torch.float64))
    for r in rs:
        r_eff = min(r, evecs.shape[1], d_h)
        V_in = evecs[:, :r_eff]
        Q_agop = _lift_input_directions(W1, V_in)
        out[f"vcr_agop_r{r}"] = _projected_vcr(Sigma_W, Sigma_B, Q_agop)

        r_p = min(r, pca_evecs.shape[1])
        Q_pca = pca_evecs[:, :r_p]
        out[f"vcr_pca_r{r}"] = _projected_vcr(Sigma_W, Sigma_B, Q_pca)

        Q_rand, _ = make_random_basis(d_h, r, seed=seed + 7000 + r)
        out[f"vcr_random_r{r}"] = _projected_vcr(Sigma_W, Sigma_B, Q_rand)
    return out
