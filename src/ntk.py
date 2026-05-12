"""Lightweight correct-logit NTK drift baseline."""

from __future__ import annotations

import torch

from .utils import EPS


def _flatten_grads(grads: tuple[torch.Tensor | None, ...]) -> torch.Tensor:
    chunks = []
    for grad in grads:
        if grad is not None:
            chunks.append(grad.detach().reshape(-1))
    if not chunks:
        return torch.empty(0)
    return torch.cat(chunks)


def compute_correct_logit_features(
    model: torch.nn.Module,
    X_probe: torch.Tensor,
    y_probe: torch.Tensor,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Rows are gradients of f_y(x) with respect to all parameters."""
    device = torch.device(device)
    model.eval()
    params = tuple(p for p in model.parameters() if p.requires_grad)
    rows = []
    for x, y in zip(X_probe, y_probe):
        model.zero_grad(set_to_none=True)
        logits = model(x.to(device).unsqueeze(0)).squeeze(0)
        target = logits[int(y.item())]
        grads = torch.autograd.grad(target, params, retain_graph=False, create_graph=False, allow_unused=True)
        rows.append(_flatten_grads(grads).cpu())
    return torch.stack(rows, dim=0).to(torch.float64)


def compute_correct_logit_ntk(
    model: torch.nn.Module,
    X_probe: torch.Tensor,
    y_probe: torch.Tensor,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    Phi = compute_correct_logit_features(model, X_probe, y_probe, device=device)
    return Phi @ Phi.T


def ntk_relative_drift(K_t: torch.Tensor, K_0: torch.Tensor) -> float:
    Kt = K_t.detach().cpu().to(torch.float64)
    K0 = K_0.detach().cpu().to(torch.float64)
    return float(torch.linalg.matrix_norm(Kt - K0).item() / (torch.linalg.matrix_norm(K0).item() + EPS))
