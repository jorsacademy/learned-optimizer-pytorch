from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def preprocess_gradient(gradient: torch.Tensor, p: float = 10.0) -> torch.Tensor:
    """Two-channel log/sign preprocessing used by early learned-optimizer work."""
    flat = gradient.reshape(-1)
    threshold = torch.exp(torch.tensor(-p, device=flat.device, dtype=flat.dtype))
    abs_grad = flat.abs()
    large = abs_grad >= threshold
    log_feature = torch.where(
        large,
        torch.log(abs_grad.clamp_min(torch.finfo(flat.dtype).tiny)) / p,
        torch.full_like(flat, -1.0),
    )
    small_scale = torch.exp(torch.tensor(p, device=flat.device, dtype=flat.dtype))
    sign_feature = torch.where(large, torch.sign(flat), small_scale * flat)
    return torch.stack((log_feature, sign_feature), dim=-1)


@dataclass
class OptimizerState:
    h: torch.Tensor
    c: torch.Tensor

    def detached(self) -> OptimizerState:
        return OptimizerState(self.h.detach(), self.c.detach())


class CoordinateWiseLSTMOptimizer(nn.Module):
    """A parameter-shared LSTM that emits one update per optimized coordinate."""

    def __init__(self, hidden_size: int = 20, update_scale: float = 0.1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.update_scale = float(update_scale)
        self.cell = nn.LSTMCell(input_size=2, hidden_size=hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def initial_state(
        self,
        n_coordinates: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> OptimizerState:
        shape = (n_coordinates, self.hidden_size)
        return OptimizerState(
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )

    def forward(
        self, gradient: torch.Tensor, state: OptimizerState
    ) -> tuple[torch.Tensor, OptimizerState]:
        features = preprocess_gradient(gradient)
        h, c = self.cell(features, (state.h, state.c))
        update = self.update_scale * torch.tanh(self.head(h).squeeze(-1))
        return update.reshape_as(gradient), OptimizerState(h, c)
