"""Input average gradient outer product (AGOP) computation."""

from __future__ import annotations

import warnings

import torch

from .metrics import effective_rank, eigengap, top_eval


def compute_input_agop(
    model: torch.nn.Module,
    X_probe: torch.Tensor,
    batch_size: int = 128,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Compute A_x = E_i J_i^T J_i for centered logits f(x_i)."""
    device = torch.device(device)
    model.eval()
    d = X_probe.shape[1]
    A = torch.zeros(d, d, device=device, dtype=torch.float64)
    n = 0

    def f_single(x: torch.Tensor) -> torch.Tensor:
        logits = model(x.unsqueeze(0)).squeeze(0)
        return logits - logits.mean()

    use_func = hasattr(torch, "func") and hasattr(torch.func, "jacrev") and hasattr(torch.func, "vmap")
    if use_func:
        jac_fn = torch.func.jacrev(f_single)

    for start in range(0, len(X_probe), int(batch_size)):
        xb = X_probe[start : start + int(batch_size)].to(device)
        if use_func:
            J = torch.func.vmap(jac_fn)(xb)  # [B, C, d]
        else:
            J_rows = []
            for x in xb:
                x_req = x.detach().clone().requires_grad_(True)
                out = f_single(x_req)
                grads = []
                for c in range(out.numel()):
                    grad = torch.autograd.grad(out[c], x_req, retain_graph=True)[0]
                    grads.append(grad)
                J_rows.append(torch.stack(grads, dim=0))
            J = torch.stack(J_rows, dim=0)
        J64 = J.to(torch.float64)
        A += torch.einsum("bcd,bce->de", J64, J64)
        n += xb.shape[0]

    A /= max(n, 1)
    A = 0.5 * (A + A.T)
    return A.detach().cpu()


def summarize_agop(A: torch.Tensor, fourier_dims: dict[int, int] | None = None) -> tuple[dict[str, float], torch.Tensor]:
    """Return scalar AGOP summaries and descending raw eigenvalues."""
    A64 = A.detach().cpu().to(torch.float64)
    symmetry_err = float(torch.linalg.matrix_norm(A64 - A64.T).item())
    evals = torch.linalg.eigvalsh(A64)
    evals = evals[torch.argsort(evals, descending=True)]
    min_eval = float(evals[-1].item()) if len(evals) else float("nan")
    if min_eval < -1e-6:
        warnings.warn(f"AGOP has a notably negative eigenvalue: {min_eval:.3e}")

    out: dict[str, float] = {
        "agop_trace": float(torch.trace(A64).item()),
        "agop_top_eval_1": top_eval(evals, 1),
        "agop_top_eval_2": top_eval(evals, 2),
        "agop_top_eval_5": top_eval(evals, 5),
        "agop_effective_rank": effective_rank(evals),
        "agop_normalized_effective_rank": effective_rank(evals) / max(A64.shape[0], 1),
        "agop_eigengap_1_2": eigengap(evals, 1, 2),
        "agop_min_eval": min_eval,
        "agop_symmetry_error": symmetry_err,
    }
    if fourier_dims:
        for K, s in fourier_dims.items():
            if len(evals) > s:
                out[f"agop_eigengap_s_splus1_K{K}"] = eigengap(evals, s, s + 1)
    return out, evals
