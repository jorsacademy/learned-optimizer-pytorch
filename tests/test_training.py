import torch

from learned_optimizer.io import load_model, save_model
from learned_optimizer.train import MetaTrainConfig, meta_train


def test_meta_training_updates_model_and_roundtrips(tmp_path):
    config = MetaTrainConfig(seed=7, episodes=6, dim=5, unroll_steps=6, hidden_size=8)
    model, metrics = meta_train(config)
    assert torch.isfinite(torch.tensor(metrics["final_meta_loss"]))
    path = tmp_path / "model.pt"
    save_model(model, path)
    loaded = load_model(path)
    gradient = torch.randn(5)
    state_a = model.initial_state(5)
    state_b = loaded.initial_state(5)
    with torch.no_grad():
        update_a, _ = model(gradient, state_a)
        update_b, _ = loaded(gradient, state_b)
    assert torch.allclose(update_a, update_b)


def test_meta_gradient_reaches_optimizer_parameters():
    from learned_optimizer.model import CoordinateWiseLSTMOptimizer
    from learned_optimizer.tasks import make_quadratic_task

    model = CoordinateWiseLSTMOptimizer(hidden_size=6)
    task = make_quadratic_task(dim=4, seed=123)
    x = torch.zeros(4, requires_grad=True)
    state = model.initial_state(4)
    loss = task.loss(x)
    gradient = torch.autograd.grad(loss, x, create_graph=True)[0]
    update, _ = model(gradient, state)
    meta_loss = task.loss(x + update)
    meta_loss.backward()
    finite_grads = [
        param.grad is not None and torch.isfinite(param.grad).all()
        for param in model.parameters()
    ]
    assert any(finite_grads)
