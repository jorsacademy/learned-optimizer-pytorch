# Learned Optimizer in PyTorch

A compact, reproducible **Learning to Optimize (L2O)** research implementation in PyTorch. The repository meta-trains a shared coordinate-wise LSTM to transform gradient histories into parameter updates, then evaluates whether that learned update rule transfers to unseen optimization problems.

The key objective is not to claim that a small learned optimizer beats Adam. The repository is designed to separate three questions that are often conflated:

1. Can an optimizer network reduce its **meta-training objective**?
2. Does it optimize **held-out problems from the same distribution**?
3. Does it generalize across **dimension and conditioning shifts**?

The CI benchmark intentionally reports negative results when the learned optimizer fails those transfer tests.

## Background

Andrychowicz et al. (NeurIPS 2016) formulate optimizer design itself as a learning problem: an LSTM consumes gradients and emits parameter updates. Wichrowska et al. (ICML 2017) emphasize that scaling and generalization are central obstacles for learned optimizers. Later large-scale systems such as VeLO investigate whether broad meta-training can produce more versatile optimizers, while independent evaluation has shown that strong general-purpose claims require careful benchmarking.

This repository is an **independent educational/research implementation**, not a reproduction of VeLO or the original Google/DeepMind codebases.

## Method

### Optimizee family

Meta-training uses batches of strongly convex quadratic objectives

\[
f(x) = \frac{1}{2} x^T A x + b^T x,
\]

where `A` is a random symmetric positive-definite matrix with controlled condition number. The exact minimizer is available from a linear solve, so suboptimality is measured without an approximate reference solver.

Training tasks use:

- dimension: `10`
- condition-number range: approximately `[1, 30]`
- randomly rotated Hessians (not diagonal-only tasks)

### Coordinate-wise LSTM optimizer

The optimizer shares one `LSTMCell` across every coordinate. At each inner optimization step it receives the classic two-channel log/sign gradient preprocessing used in early learned-optimizer work and produces a bounded update.

Because parameters share the same recurrent rule, the model can be evaluated on dimensions it never saw during meta-training.

### Meta-objective

For each optimizee, the repository unrolls the learned optimizer differentiably and minimizes the average `log(1 + relative_suboptimality)` over the second half of the inner trajectory.

The outer optimizer is PyTorch Adam. Gradients through the complete inner optimization trajectory are clipped before the outer update.

## Baselines

Every held-out problem receives the same optimization-step budget. The repository compares:

- learned coordinate-wise LSTM
- SGD
- Momentum
- Adam

These baseline hyperparameters are fixed globally rather than tuned separately on the held-out splits. PyTorch's Adam documentation remains the reference for the hand-designed adaptive baseline semantics.

## Generalization splits

`benchmark.py` evaluates three deterministic suites:

| Split | Dimension | Condition range | Purpose |
|---|---:|---:|---|
| in-distribution | 10 | 1–30 | ordinary held-out transfer |
| dimension shift | 25 | 1–30 | coordinate-wise scale transfer |
| conditioning shift | 10 | 30–100 | landscape-distribution shift |

Metrics are normalized by each task's initial exact suboptimality:

- mean final relative gap
- median final relative gap
- mean log10 trajectory area (`log_auc_mean`)

Lower is better. A relative gap of `0.01` means the remaining objective error is 1% of the initial error.

## Reproducible smoke study

```bash
python -m pip install -e ".[dev]"

python scripts/train.py \
  --seed 42 \
  --outer-steps 60 \
  --inner-steps 16 \
  --batch-size 12 \
  --dimension 10 \
  --hidden-size 16 \
  --out artifacts/learned_optimizer.pt \
  --metrics-out artifacts/training_metrics.json

python scripts/benchmark.py \
  --model artifacts/learned_optimizer.pt \
  --seed 9000 \
  --steps 32 \
  --output artifacts/benchmark.json
```

### Current local smoke result

Before publication, the same configuration used by CI produced:

- first outer meta-loss: `0.6950`
- final outer meta-loss: `0.3334`
- best outer meta-loss: `0.3262`

So the optimizer **does learn its meta-training objective**. That is not enough to establish optimizer quality.

Held-out final relative-gap means were:

| Split | Learned LSTM | SGD | Momentum | Adam |
|---|---:|---:|---:|---:|
| in-distribution | 0.865 | **0.014** | 0.017 | 0.102 |
| dimension shift | 0.679 | **0.015** | 0.017 | 0.091 |
| conditioning shift | 1.726 | diverged badly | **0.020** | 0.087 |

The learned optimizer is therefore **not competitive under this small meta-training budget**. The conditioning-shift result is especially important: the learned rule reduces meta-training loss but does not robustly transfer to a harder curvature regime.

This negative result is intentional and should not be replaced with a claim such as "the learned optimizer beats Adam." GitHub Actions reruns the study and preserves the JSON artifacts.

## Why keep a negative result?

Learned optimizers are meta-learned dynamical systems. Low outer loss can coexist with poor transfer because the optimizer can over-specialize to the optimizee distribution, unroll horizon, scale statistics, or curvature range seen during training. That makes explicit distribution-shift tests more informative than reporting only training curves.

The repository therefore treats generalization failure as a result rather than a CI failure.

## Tests

The test suite checks:

- exact quadratic optimum / near-zero gradient
- finite log/sign gradient preprocessing
- coordinate-wise state and output shapes
- differentiability through the optimizer update
- model serialization round-trip
- differentiable multi-step unrolling
- finite SGD/Momentum/Adam trajectories
- deterministic meta-training parameter updates
- existence of all held-out benchmark splits and baselines

## GitHub Actions

CI runs on Ubuntu 24.04 with Python 3.10, 3.11 and 3.12.

It performs:

1. CPU-only PyTorch installation
2. Ruff linting
3. unit tests
4. deterministic meta-training smoke
5. held-out benchmark suite
6. artifact upload of the trained optimizer and JSON metrics

The smoke job is deliberately small enough for ordinary GitHub-hosted runners. It is a reproducibility/integration test, not a claim of state-of-the-art learned-optimizer performance.

## Research extensions

Useful next experiments include:

- longer outer training with truncated backpropagation through time
- curriculum over optimizee dimensions and condition numbers
- richer per-coordinate state (moments, time features, parameter statistics)
- hierarchical/global recurrent state as in scale-generalizing learned optimizers
- evolution-strategy or persistent-evolution-strategy outer-gradient estimators
- neural-network optimizees instead of only analytic quadratics
- tuned baseline sweeps under equal meta-compute and wall-clock budgets
- validation of wall-clock overhead, not only inner-step quality

## References

- Andrychowicz et al. (2016), *Learning to learn by gradient descent by gradient descent*, NeurIPS 29.
- Wichrowska et al. (2017), *Learned Optimizers that Scale and Generalize*, ICML / PMLR 70.
- Metz et al. (2019), *Understanding and correcting pathologies in the training of learned optimizers*, ICML / PMLR 97.
- Metz et al. (2022), *VeLO: Training Versatile Learned Optimizers by Scaling Up*.
- Yang et al. (2023), *Learning to Generalize Provably in Learning to Optimize*, AISTATS / PMLR 206.
- Rezk et al. (2023), *Is Scaling Learned Optimizers Worth It? Evaluating The Value of VeLO's 4000 TPU Months*.
- PyTorch `torch.optim.Adam` documentation for the adaptive baseline.

See `RESEARCH_NOTES.md` for methodological decisions and limitations.

## License

MIT.
