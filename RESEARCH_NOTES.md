# Research Notes

## Positioning

This repository studies the "learning to optimize" paradigm, where the update rule itself is parameterized and meta-trained. It follows the broad coordinate-wise recurrent optimizer idea introduced by Andrychowicz et al. (2016), while intentionally keeping the task family and implementation small enough to inspect and execute in CI.

It is not a reproduction of the paper's full experiments or later hierarchical learned optimizers.

## Why quadratics first

Strongly convex quadratics provide known optima, controllable conditioning and deterministic task generation. They make it possible to separate implementation correctness from benchmark rhetoric. Random orthogonal eigenspaces prevent the benchmark from collapsing to a trivial axis-aligned step-size problem.

## Meta-objective

Raw losses differ substantially across generated tasks. Optimizing the unnormalized trajectory average made episode scale dominate the outer objective in an early local experiment. The final implementation normalizes each trajectory loss by the detached initial task loss. This improved meta-training stability but did not make the learned optimizer universally superior.

## Higher-order differentiation

The inner gradient is computed with `torch.autograd.grad(..., create_graph=True)`. Without the derivative graph, the outer objective cannot propagate through the inner gradient calculation to the LSTM parameters. A dedicated unit test checks that finite gradients reach learned-optimizer parameters.

## Baseline fairness

A learned optimizer has already consumed a task distribution during meta-training. Comparing it to an arbitrarily chosen SGD/Adam learning rate would be weak methodology. The benchmark therefore creates separate baseline-tuning tasks for each regime, selects the best learning rate from a fixed grid by median final loss, freezes it, and only then evaluates untouched benchmark seeds.

The learned optimizer is not tuned on benchmark tasks.

## Generalization tests

- ID quadratics test new seeds from the same curvature family.
- OOD quadratics extend the curvature range from about 1--20 to 1--100.
- Rosenbrock changes the objective family entirely.

These are intentionally progressively harder transfer settings.

## Interpretation

Do not interpret a win over Adam on one finite-horizon suite as a universal optimizer improvement. Likewise, a loss to tuned SGD is scientifically useful: it exposes how much inductive bias a simple classical method has on smooth convex quadratics.

The main reproducible contribution is the complete experimental loop: task generation, differentiable unroll, meta-training, baseline tuning on separate tasks, held-out/OOD evaluation, tests and CI artifacts.
