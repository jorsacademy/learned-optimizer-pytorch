# Learned Optimizer in PyTorch

A reproducible research implementation of a **coordinate-wise LSTM optimizer** trained by meta-gradient descent. The repository asks a narrow question: can an optimizer learned on a distribution of small convex quadratic problems transfer to unseen optimization tasks, and how does it compare with separately tuned SGD and Adam?

This is an independent educational/research implementation inspired by the learned-optimizer literature. It is **not** a reproduction claim for any published benchmark.

## Method

The optimizee is a parameter vector `x`. At every optimization step, the learned optimizer receives a two-channel preprocessing of each coordinate gradient and shares one LSTMCell across coordinates. The LSTM emits one bounded update per coordinate.

Meta-training differentiates through the unrolled optimization trajectory with:

```python
gradient = torch.autograd.grad(loss, x, create_graph=True)[0]
update, state = learned_optimizer(gradient, state)
x = x + update
```

`create_graph=True` is essential because the outer/meta objective must differentiate through the inner gradient computation. PyTorch documents this as constructing the derivative graph for higher-order derivatives.

To prevent tasks with large raw objective scales from dominating meta-training, each episode minimizes trajectory losses normalized by that task's detached initial loss. Gradient norm clipping is applied to the learned optimizer parameters.

## Gradient preprocessing

The implementation uses the two-feature log/sign style preprocessing popularized in early learned-optimizer work. For sufficiently large gradient magnitude, the features are approximately `log(|g|)/p` and `sign(g)`; very small gradients use a linear fallback.

## Training distribution

Meta-training tasks are seeded strongly convex quadratics

`0.5 * (x - x*)^T Q (x - x*)`

with random orthogonal eigenvectors and log-spaced positive eigenvalues. The default training condition-number range is approximately 1--20. Ground-truth optimum is known exactly (`x*`), although the optimizer receives only gradients.

## Validation protocol

Three disjoint evaluation regimes are used:

1. **In-distribution quadratics**: unseen seeds, training-like curvature range.
2. **OOD quadratics**: unseen seeds with curvature range extended to about 1--100.
3. **Rosenbrock transfer**: a non-quadratic objective never seen during meta-training.

SGD and Adam are not given hand-picked test learning rates. For each regime, their learning rates are selected on a **separate tuning task set** from a fixed candidate grid, then frozen before the untouched benchmark set is evaluated. The learned optimizer receives no test-set tuning.

Both final loss and mean log10 trajectory loss are reported. This avoids judging an optimizer only by its last iterate.

## Reproduce

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest

python scripts/train.py \
  --seed 42 --episodes 220 --dim 10 --unroll-steps 20 --hidden-size 20 \
  --out artifacts/learned_optimizer.pt \
  --metrics-out artifacts/training_metrics.json

python scripts/benchmark.py \
  --model artifacts/learned_optimizer.pt --seed 9000 \
  --tasks 8 --dim 10 --steps 30 \
  --output artifacts/benchmark.json
```

## What the smoke study is designed to test

The experiment is deliberately capable of producing a negative result. It does **not** fail CI merely because the learned optimizer loses to SGD or Adam. CI fails for code correctness problems; the benchmark artifact records empirical performance as observed.

In local deterministic validation before publication, the learned optimizer showed partial transfer: it was competitive with Adam on some quadratic regimes and improved substantially over Adam on the tested OOD quadratic set, but tuned SGD remained much stronger. Rosenbrock transfer also did not dominate tuned SGD. The exact GitHub-hosted smoke metrics are stored as Actions artifacts and should be treated as the authoritative run for the current commit.

This is an important limitation rather than a defect to hide: learned optimizers can overfit their task distribution, optimization horizon, parameterization and loss geometry. Generalization is one of the central difficulties in this field.

## Tests

The test suite covers:

- positive-definite task construction and known quadratic optimum;
- Rosenbrock optimum;
- finite gradient preprocessing and state/update shapes;
- SGD and Adam sanity checks on a convex problem;
- meta-training execution and model serialization round-trip;
- explicit verification that meta-gradients reach learned-optimizer parameters;
- benchmark schema containing learned, Adam and SGD results for ID/OOD/transfer regimes.

GitHub Actions runs Python 3.10, 3.11 and 3.12 on CPU, then performs a deterministic meta-training + benchmark smoke and uploads the trained model and JSON results.

## Scientific scope and limitations

- The coordinate-wise LSTM has parameter sharing but no explicit cross-coordinate communication.
- Small quadratic tasks are useful for transparent validation but are not a claim about large neural-network training.
- Unrolled meta-gradients are memory/computation intensive; this repository uses short horizons suitable for CI.
- Baseline hyperparameter selection is separated from test evaluation, but the candidate grids are still finite.
- Results depend on task distribution and horizon. A learned optimizer that wins one regime should not be assumed to generalize universally.
- No claim is made that this architecture matches modern large-scale learned optimizers.

## Primary references

- Andrychowicz et al. (2016), *Learning to learn by gradient descent by gradient descent*.
- Wichrowska et al. (2017), *Learned Optimizers that Scale and Generalize*.
- PyTorch `torch.autograd.grad` documentation for higher-order differentiation via `create_graph=True`.

See `RESEARCH_NOTES.md` for design decisions and interpretation guidance.

## License

This repository is licensed under the **JORS Academy Non-Commercial Source License 1.0**. Commercial use is prohibited without a separate prior written commercial license. See [`LICENSE`](LICENSE) for the complete terms.
