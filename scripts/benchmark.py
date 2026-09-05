from __future__ import annotations

import argparse
import json
from pathlib import Path

from learned_optimizer.benchmark import benchmark_to_dict, evaluate_suite
from learned_optimizer.io import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/learned_optimizer.pt")
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output", default="artifacts/benchmark.json")
    args = parser.parse_args()

    model = load_model(args.model)
    summary = evaluate_suite(
        model,
        seed=args.seed,
        tasks=args.tasks,
        dim=args.dim,
        steps=args.steps,
    )
    payload = benchmark_to_dict(summary)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
