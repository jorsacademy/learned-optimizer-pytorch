from __future__ import annotations

from pathlib import Path

import torch

from .model import CoordinateWiseLSTMOptimizer


def save_model(model: CoordinateWiseLSTMOptimizer, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "hidden_size": model.hidden_size,
            "update_scale": model.update_scale,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_model(path: str | Path) -> CoordinateWiseLSTMOptimizer:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = CoordinateWiseLSTMOptimizer(
        hidden_size=int(payload["hidden_size"]),
        update_scale=float(payload["update_scale"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
