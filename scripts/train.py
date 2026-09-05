from __future__ import annotations

import argparse
import json
from pathlib import Path

from learned_optimizer.io import save_model
from learned_optimizer.train import MetaTrainConfig, meta_train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--unroll-steps", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=20)
    parser.add_argument("--out", default="artifacts/learned_optimizer.pt")
    parser.add_argument("--metrics-out", default="artifacts/training_metrics.json")
    args = parser.parse_args()

    config = MetaTrainConfig(
        seed=args.seed,
        episodes=args.episodes,
        dim=args.dim,
        unroll_steps=args.unroll_steps,
        hidden_size=args.hidden_size,
    )
    model, metrics = meta_train(config)
    save_model(model, args.out)
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
