from learned_optimizer.benchmark import evaluate_suite
from learned_optimizer.train import MetaTrainConfig, meta_train


def test_benchmark_has_id_and_ood_baselines():
    model, _ = meta_train(MetaTrainConfig(seed=5, episodes=4, dim=4, unroll_steps=4, hidden_size=6))
    result = evaluate_suite(model, seed=200, tasks=2, dim=4, steps=4)
    for suite in (result.in_distribution, result.ood_quadratic, result.rosenbrock):
        assert set(suite) == {"learned_lstm", "adam", "sgd"}
        for metrics in suite.values():
            assert metrics["median_final_loss"] >= 0.0
