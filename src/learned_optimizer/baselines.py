from __future__ import annotations

from collections.abc import Callable

import torch

LossFn = Callable[[torch.Tensor], torch.Tensor]


def optimize_sgd(loss_fn: LossFn, x0: torch.Tensor, *, steps: int, lr: float) -> list[float]:
    x = torch.nn.Parameter(x0.detach().clone())
    optimizer = torch.optim.SGD([x], lr=lr)
    history: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(x)
        history.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    history.append(float(loss_fn(x).detach()))
    return history


def optimize_adam(loss_fn: LossFn, x0: torch.Tensor, *, steps: int, lr: float) -> list[float]:
    x = torch.nn.Parameter(x0.detach().clone())
    optimizer = torch.optim.Adam([x], lr=lr)
    history: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(x)
        history.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    history.append(float(loss_fn(x).detach()))
    return history
