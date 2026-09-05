from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from .model import CoordinateWiseLSTMOptimizer
from .tasks import make_quadratic_task


@dataclass(frozen=True)
class MetaTrainConfig:
    seed: int = 42
    episodes: int = 80
    dim: int = 10
    unroll_steps: int = 20
    hidden_size: int = 20
    meta_lr: float = 3e-3
    gradient_clip: float = 1.0
    condition_min: float = 1.0
    condition_max: float = 20.0


def _initial_point(dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return 3.0 * torch.randn(dim, generator=generator)


def meta_train(
    config: MetaTrainConfig,
) -> tuple[CoordinateWiseLSTMOptimizer, dict[str, float | int | dict[str, object]]]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = CoordinateWiseLSTMOptimizer(hidden_size=config.hidden_size)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=config.meta_lr)
    episode_losses: list[float] = []

    for episode in range(config.episodes):
        task_seed = config.seed + 10_000 + episode
        task = make_quadratic_task(
            dim=config.dim,
            condition_min=config.condition_min,
            condition_max=config.condition_max,
            seed=task_seed,
        )
        x = _initial_point(config.dim, config.seed + 20_000 + episode).requires_grad_(True)
        state = model.initial_state(config.dim, dtype=x.dtype)
        trajectory_losses: list[torch.Tensor] = []
        initial_loss = task.loss(x).detach().clamp_min(1e-8)

        for _ in range(config.unroll_steps):
            loss = task.loss(x)
            gradient = torch.autograd.grad(loss, x, create_graph=True)[0]
            update, state = model(gradient, state)
            x = x + update
            trajectory_losses.append(task.loss(x) / initial_loss)

        meta_loss = torch.stack(trajectory_losses).mean()
        meta_optimizer.zero_grad(set_to_none=True)
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        meta_optimizer.step()
        episode_losses.append(float(meta_loss.detach()))

    tail = episode_losses[max(0, len(episode_losses) - 10) :]
    metrics: dict[str, float | int | dict[str, object]] = {
        "initial_meta_loss": episode_losses[0],
        "final_meta_loss": episode_losses[-1],
        "tail_mean_meta_loss": float(np.mean(tail)),
        "episodes": config.episodes,
        "config": asdict(config),
    }
    return model, metrics
