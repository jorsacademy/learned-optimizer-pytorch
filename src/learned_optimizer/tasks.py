from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuadraticTask:
    q: torch.Tensor
    target: torch.Tensor

    @property
    def dim(self) -> int:
        return int(self.target.numel())

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        diff = x - self.target
        return 0.5 * diff @ self.q @ diff

    @property
    def optimum(self) -> float:
        return 0.0


def make_quadratic_task(
    *,
    dim: int,
    condition_min: float = 1.0,
    condition_max: float = 20.0,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> QuadraticTask:
    if dim < 2:
        raise ValueError("dim must be at least 2")
    if condition_min <= 0 or condition_max < condition_min:
        raise ValueError("invalid curvature range")
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    raw = torch.randn(dim, dim, generator=g, dtype=dtype)
    orthogonal, _ = torch.linalg.qr(raw)
    log_min = torch.log(torch.tensor(condition_min, dtype=dtype))
    log_max = torch.log(torch.tensor(condition_max, dtype=dtype))
    eig = torch.exp(torch.linspace(log_min, log_max, dim, dtype=dtype))
    q = orthogonal @ torch.diag(eig) @ orthogonal.T
    target = 2.0 * torch.randn(dim, generator=g, dtype=dtype)
    return QuadraticTask(q=q, target=target)


def rosenbrock_loss(x: torch.Tensor) -> torch.Tensor:
    if x.numel() < 2:
        raise ValueError("Rosenbrock requires at least two coordinates")
    left = x[:-1]
    right = x[1:]
    return torch.sum(100.0 * (right - left.square()).square() + (1.0 - left).square())
