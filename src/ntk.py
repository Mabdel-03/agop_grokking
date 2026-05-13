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


def _named_linear_blocks(model: torch.nn.Module) -> list[tuple[str, torch.nn.Linear]]:
    return [(name, mod) for name, mod in model.named_modules() if isinstance(mod, torch.nn.Linear)]


def compute_block_backward_gram(
    model: torch.nn.Module,
    X_probe: torch.Tensor,
    y_probe: torch.Tensor,
    device: torch.device | str = "cuda",
) -> dict[str, torch.Tensor]:
    """Per-block backward Gram B_l = H_l H_l^T from correct-logit weight gradients."""
    device = torch.device(device)
    model.eval()
    blocks = _named_linear_blocks(model)
    rows: dict[str, list[torch.Tensor]] = {name: [] for name, _ in blocks}
    for x, y in zip(X_probe, y_probe):
        model.zero_grad(set_to_none=True)
        logits = model(x.to(device).unsqueeze(0)).squeeze(0)
        target = logits[int(y.item())]
        weight_params = [mod.weight for _, mod in blocks]
        grads = torch.autograd.grad(target, weight_params, retain_graph=False, allow_unused=True)
        for (name, _), g in zip(blocks, grads):
            if g is None:
                rows[name].append(torch.zeros(0, dtype=torch.float64))
            else:
                rows[name].append(g.detach().reshape(-1).cpu().to(torch.float64))
    out: dict[str, torch.Tensor] = {}
    for name, _ in blocks:
        H = torch.stack(rows[name], dim=0)
        out[name] = H @ H.T
    return out


def compute_block_forward_gram(
    model: torch.nn.Module,
    X_probe: torch.Tensor,
    device: torch.device | str = "cuda",
) -> dict[str, torch.Tensor]:
    """Per-block forward Gram G_l(i,j) = <phi_l(x_i), phi_l(x_j)> for each linear input."""
    device = torch.device(device)
    model.eval()
    blocks = _named_linear_blocks(model)
    captures: dict[str, list[torch.Tensor]] = {name: [] for name, _ in blocks}
    handles = []
    for name, mod in blocks:
        def _hook_factory(n):
            def _hook(_module, inputs, _output):
                phi = inputs[0].detach()
                captures[n].append(phi.reshape(phi.shape[0], -1).cpu().to(torch.float64))
            return _hook
        handles.append(mod.register_forward_hook(_hook_factory(name)))
    try:
        with torch.no_grad():
            _ = model(X_probe.to(device))
    finally:
        for h in handles:
            h.remove()
    out: dict[str, torch.Tensor] = {}
    for name, _ in blocks:
        phi = torch.cat(captures[name], dim=0)
        out[name] = phi @ phi.T
    return out
