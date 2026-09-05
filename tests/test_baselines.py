import torch

from learned_optimizer.baselines import optimize_adam, optimize_sgd


def loss_fn(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum((x - 2.0) ** 2)


def test_adam_reduces_convex_loss():
    history = optimize_adam(loss_fn, torch.zeros(5), steps=40, lr=0.15)
    assert history[-1] < history[0] * 0.02


def test_sgd_reduces_convex_loss():
    history = optimize_sgd(loss_fn, torch.zeros(5), steps=40, lr=0.1)
    assert history[-1] < history[0] * 0.01
