import torch

from learned_optimizer.model import CoordinateWiseLSTMOptimizer, preprocess_gradient


def test_gradient_preprocessing_shape_and_finite():
    g = torch.tensor([0.0, 1e-8, -0.2, 3.0])
    features = preprocess_gradient(g)
    assert features.shape == (4, 2)
    assert torch.isfinite(features).all()


def test_coordinatewise_optimizer_shapes():
    model = CoordinateWiseLSTMOptimizer(hidden_size=8)
    gradient = torch.randn(7)
    state = model.initial_state(7)
    update, next_state = model(gradient, state)
    assert update.shape == gradient.shape
    assert next_state.h.shape == (7, 8)
