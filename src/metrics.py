"""Metric helpers for training dynamics and AGOP spectra."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F

from .utils import EPS


def make_checkpoint_steps(max_steps: int, preset: str = "main") -> list[int]:
    """Construct checkpoint schedules with denser early measurements."""
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    preset = (preset or "main").lower()
    steps = {0, int(max_steps)}
    if preset == "smoke":
        intervals = [(50, 10), (200, 25), (max_steps, 50)]
    elif preset == "pilot_scout":
        intervals = [(500, 100), (2000, 250), (max_steps, 500)]
    elif preset == "pilot_metrics":
        intervals = [(200, 50), (2000, 250), (10000, 1000), (max_steps, 2000)]
    else:
        intervals = [(200, 10), (2000, 50), (10000, 200), (max_steps, 1000)]

    last = 0
    for stop, stride in intervals:
        stop = min(int(stop), max_steps)
        for step in range(last, stop + 1, int(stride)):
            steps.add(step)
        last = max(last, stop)
    return sorted(s for s in steps if 0 <= s <= max_steps)


@torch.no_grad()
def evaluate_loss_acc(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor, device: torch.device) -> tuple[float, float]:
    model.eval()
    Xd = X.to(device)
    yd = y.to(device)
    logits = model(Xd)
    loss = F.cross_entropy(logits, yd)
    acc = (logits.argmax(dim=-1) == yd).float().mean()
    return float(loss.item()), float(acc.item())


def parameter_l2_norm(model: torch.nn.Module) -> float:
    total = 0.0
    with torch.no_grad():
        for param in model.parameters():
            total += float(param.detach().pow(2).sum().item())
    return math.sqrt(total)


def effective_rank(eigenvalues: torch.Tensor) -> float:
    vals = eigenvalues.detach().cpu().to(torch.float64).clamp_min(0)
    total = vals.sum()
    if float(total.item()) <= 0:
        return 0.0
    probs = vals / total
    entropy = -(probs * torch.log(probs + EPS)).sum()
    return float(torch.exp(entropy).item())


def top_eval(eigenvalues: torch.Tensor, idx_one_based: int) -> float:
    if len(eigenvalues) < idx_one_based:
        return float("nan")
    return float(eigenvalues[idx_one_based - 1].item())


def eigengap(eigenvalues: torch.Tensor, i_one_based: int, j_one_based: int) -> float:
    if len(eigenvalues) < max(i_one_based, j_one_based):
        return float("nan")
    return float((eigenvalues[i_one_based - 1] - eigenvalues[j_one_based - 1]).item())


def first_crossing(
    steps: Iterable[int],
    values: Iterable[float],
    threshold: float,
    direction: str = "above",
    consecutive: int = 1,
) -> int | None:
    steps_list = list(steps)
    vals = list(values)
    good = []
    for v in vals:
        if v is None or math.isnan(float(v)):
            good.append(False)
        elif direction == "above":
            good.append(float(v) >= threshold)
        elif direction == "below":
            good.append(float(v) <= threshold)
        else:
            raise ValueError(f"direction must be above/below, got {direction}")
    for i in range(len(good)):
        if all(good[i : i + consecutive]) and len(good[i : i + consecutive]) == consecutive:
            return int(steps_list[i])
    if consecutive > 1:
        return first_crossing(steps_list, vals, threshold, direction=direction, consecutive=1)
    return None
