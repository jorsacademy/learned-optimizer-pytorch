import torch

from learned_optimizer.tasks import make_quadratic_task, rosenbrock_loss


def test_quadratic_has_zero_loss_at_target():
    task = make_quadratic_task(dim=6, seed=1)
    assert torch.isclose(task.loss(task.target), torch.tensor(0.0))
    eig = torch.linalg.eigvalsh(task.q)
    assert torch.all(eig > 0)


def test_rosenbrock_optimum():
    x = torch.ones(5)
    assert torch.isclose(rosenbrock_loss(x), torch.tensor(0.0))
