from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .baselines import optimize_adam, optimize_sgd
from .model import CoordinateWiseLSTMOptimizer
from .tasks import make_quadratic_task, rosenbrock_loss

LossFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class MethodResult:
    final_loss: float
    auc_log10_loss: float


@dataclass(frozen=True)
class BenchmarkSummary:
    in_distribution: dict[str, dict[str, float]]
    ood_quadratic: dict[str, dict[str, float]]
    rosenbrock: dict[str, dict[str, float]]
    selected_learning_rates: dict[str, dict[str, float]]
    config: dict[str, int | float]


def _learned_history(
    model: CoordinateWiseLSTMOptimizer,
    loss_fn: LossFn,
    x0: torch.Tensor,
    *,
    steps: int,
) -> list[float]:
    x = x0.detach().clone().requires_grad_(True)
    state = model.initial_state(x.numel(), dtype=x.dtype)
    history: list[float] = []
    for _ in range(steps):
        loss = loss_fn(x)
        history.append(float(loss.detach()))
        gradient = torch.autograd.grad(loss, x, create_graph=False)[0]
        with torch.no_grad():
            update, state = model(gradient, state)
        x = (x + update).detach().requires_grad_(True)
        state = state.detached()
    history.append(float(loss_fn(x).detach()))
    return history


def _score(history: list[float]) -> MethodResult:
    values = np.asarray(history, dtype=float)
    safe = np.clip(values, 1e-12, None)
    return MethodResult(
        final_loss=float(values[-1]),
        auc_log10_loss=float(np.mean(np.log10(safe))),
    )


def _aggregate(rows: dict[str, list[MethodResult]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for method, results in rows.items():
        final = np.array([item.final_loss for item in results], dtype=float)
        auc = np.array([item.auc_log10_loss for item in results], dtype=float)
        output[method] = {
            "median_final_loss": float(np.median(final)),
            "mean_final_loss": float(np.mean(final)),
            "median_auc_log10_loss": float(np.median(auc)),
        }
    return output


def _initial_point(dim: int, seed: int, *, rosenbrock: bool = False) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if rosenbrock:
        return -1.5 + 0.35 * torch.randn(dim, generator=generator)
    return 3.0 * torch.randn(dim, generator=generator)


def _tune_lr(
    method: str,
    task_specs: list[tuple[LossFn, torch.Tensor]],
    *,
    steps: int,
    candidates: tuple[float, ...],
) -> float:
    scores: list[tuple[float, float]] = []
    for lr in candidates:
        final_losses = []
        for loss_fn, x0 in task_specs:
            if method == "adam":
                history = optimize_adam(loss_fn, x0, steps=steps, lr=lr)
            elif method == "sgd":
                history = optimize_sgd(loss_fn, x0, steps=steps, lr=lr)
            else:
                raise ValueError(f"unknown method: {method}")
            final_losses.append(history[-1])
        scores.append((float(np.median(final_losses)), lr))
    scores.sort(key=lambda item: (item[0], item[1]))
    return float(scores[0][1])


def _quadratic_specs(
    *,
    seeds: list[int],
    dim: int,
    condition_min: float,
    condition_max: float,
) -> list[tuple[LossFn, torch.Tensor]]:
    specs = []
    for seed in seeds:
        task = make_quadratic_task(
            dim=dim,
            condition_min=condition_min,
            condition_max=condition_max,
            seed=seed,
        )
        specs.append((task.loss, _initial_point(dim, seed + 777)))
    return specs


def _rosenbrock_specs(*, seeds: list[int], dim: int) -> list[tuple[LossFn, torch.Tensor]]:
    return [(rosenbrock_loss, _initial_point(dim, seed, rosenbrock=True)) for seed in seeds]


def _compare_specs(
    model: CoordinateWiseLSTMOptimizer,
    specs: list[tuple[LossFn, torch.Tensor]],
    *,
    steps: int,
    sgd_lr: float,
    adam_lr: float,
) -> dict[str, dict[str, float]]:
    rows: dict[str, list[MethodResult]] = {"learned_lstm": [], "adam": [], "sgd": []}
    for loss_fn, x0 in specs:
        rows["learned_lstm"].append(
            _score(_learned_history(model, loss_fn, x0, steps=steps))
        )
        rows["adam"].append(_score(optimize_adam(loss_fn, x0, steps=steps, lr=adam_lr)))
        rows["sgd"].append(_score(optimize_sgd(loss_fn, x0, steps=steps, lr=sgd_lr)))
    return _aggregate(rows)


def evaluate_suite(
    model: CoordinateWiseLSTMOptimizer,
    *,
    seed: int = 9000,
    tasks: int = 8,
    dim: int = 10,
    steps: int = 30,
) -> BenchmarkSummary:
    model.eval()
    n_tune = max(4, tasks // 2)
    tune_id = _quadratic_specs(
        seeds=[seed - 1000 + i for i in range(n_tune)],
        dim=dim,
        condition_min=1.0,
        condition_max=20.0,
    )
    tune_ood = _quadratic_specs(
        seeds=[seed - 2000 + i for i in range(n_tune)],
        dim=dim,
        condition_min=1.0,
        condition_max=100.0,
    )
    tune_rb = _rosenbrock_specs(
        seeds=[seed - 3000 + i for i in range(n_tune)],
        dim=dim,
    )
    learning_rates = {
        "in_distribution": {
            "adam": _tune_lr(
                "adam", tune_id, steps=steps, candidates=(0.01, 0.03, 0.05, 0.1, 0.2)
            ),
            "sgd": _tune_lr(
                "sgd", tune_id, steps=steps, candidates=(0.005, 0.01, 0.02, 0.05, 0.1)
            ),
        },
        "ood_quadratic": {
            "adam": _tune_lr(
                "adam", tune_ood, steps=steps, candidates=(0.005, 0.01, 0.03, 0.05, 0.1)
            ),
            "sgd": _tune_lr(
                "sgd", tune_ood, steps=steps, candidates=(0.001, 0.003, 0.005, 0.01, 0.02)
            ),
        },
        "rosenbrock": {
            "adam": _tune_lr(
                "adam", tune_rb, steps=steps, candidates=(0.001, 0.003, 0.005, 0.01, 0.02)
            ),
            "sgd": _tune_lr(
                "sgd", tune_rb, steps=steps, candidates=(0.0001, 0.0003, 0.0005, 0.001)
            ),
        },
    }

    id_specs = _quadratic_specs(
        seeds=[seed + i for i in range(tasks)],
        dim=dim,
        condition_min=1.0,
        condition_max=20.0,
    )
    ood_specs = _quadratic_specs(
        seeds=[seed + 1000 + i for i in range(tasks)],
        dim=dim,
        condition_min=1.0,
        condition_max=100.0,
    )
    rb_specs = _rosenbrock_specs(
        seeds=[seed + 2000 + i for i in range(max(4, tasks // 2))],
        dim=dim,
    )

    return BenchmarkSummary(
        in_distribution=_compare_specs(
            model,
            id_specs,
            steps=steps,
            sgd_lr=learning_rates["in_distribution"]["sgd"],
            adam_lr=learning_rates["in_distribution"]["adam"],
        ),
        ood_quadratic=_compare_specs(
            model,
            ood_specs,
            steps=steps,
            sgd_lr=learning_rates["ood_quadratic"]["sgd"],
            adam_lr=learning_rates["ood_quadratic"]["adam"],
        ),
        rosenbrock=_compare_specs(
            model,
            rb_specs,
            steps=steps,
            sgd_lr=learning_rates["rosenbrock"]["sgd"],
            adam_lr=learning_rates["rosenbrock"]["adam"],
        ),
        selected_learning_rates=learning_rates,
        config={"seed": seed, "tasks": tasks, "dim": dim, "steps": steps},
    )


def benchmark_to_dict(summary: BenchmarkSummary) -> dict[str, object]:
    return asdict(summary)
