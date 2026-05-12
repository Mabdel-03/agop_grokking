"""Small MLPs for modular addition grokking."""

from __future__ import annotations

import math

import torch
from torch import nn


class QuadraticMLP(nn.Module):
    """Two-layer MLP with squared hidden preactivations."""

    def __init__(self, input_dim: int, hidden_width: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_width, bias=bias)
        self.linear2 = nn.Linear(hidden_width, output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear1(x)
        h = h.square()
        return self.linear2(h)


class ReLUMLP(nn.Module):
    """Two-layer ReLU MLP fallback model."""

    def __init__(self, input_dim: int, hidden_width: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_width, bias=bias)
        self.linear2 = nn.Linear(hidden_width, output_dim, bias=bias)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.linear1(x))
        return self.linear2(h)


def init_scaled_normal(model: nn.Module, init_scale: float, bias_scale: float = 0.0) -> None:
    """Initialize linear weights as N(0, init_scale/sqrt(fan_in))."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            fan_in = module.weight.shape[1]
            nn.init.normal_(module.weight, mean=0.0, std=float(init_scale) / math.sqrt(fan_in))
            if module.bias is not None:
                if bias_scale > 0:
                    nn.init.normal_(module.bias, mean=0.0, std=float(bias_scale))
                else:
                    nn.init.zeros_(module.bias)


def make_model(
    model_type: str,
    input_dim: int,
    hidden_width: int,
    output_dim: int,
    init_scale: float,
    seed: int,
    bias: bool = True,
) -> nn.Module:
    torch.manual_seed(seed)
    model_type = model_type.lower()
    if model_type in {"quadratic", "quadratic_mlp", "quad"}:
        model = QuadraticMLP(input_dim, hidden_width, output_dim, bias=bias)
    elif model_type in {"relu", "relu_mlp"}:
        model = ReLUMLP(input_dim, hidden_width, output_dim, bias=bias)
    else:
        raise ValueError(f"Unknown model_type={model_type!r}")
    init_scaled_normal(model, init_scale=init_scale)
    return model
