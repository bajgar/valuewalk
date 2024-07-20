import torch
import pyro.contrib.gp as gp
import pytest

from models.gp.gp_prior import GPPrior


def test_log_prob():
    x = torch.linspace(0, 1, 10)
    y = torch.sin(x * (2 * 3.14))
    prior_mean_value = 0.0
    lengthscale = 0.1
    scale = 1.0

    mean_function = lambda x: prior_mean_value * torch.ones(x.size(0))
    kernel = gp.kernels.RBF(input_dim=1, variance=torch.tensor(scale), lengthscale=torch.tensor(lengthscale))
    gp_prior = GPPrior(mean_function=mean_function, kernel=kernel)

    log_prob = gp_prior.log_prob(x, y)

    assert isinstance(log_prob, torch.Tensor)
    assert log_prob.size() == torch.Size([])


def test_default_mean_and_kernel():
    x = torch.linspace(0, 1, 10)
    y = torch.sin(x * (2 * 3.14))
    gp_prior_default = GPPrior()
    log_prob = gp_prior_default.log_prob(x, y)

    assert isinstance(log_prob, torch.Tensor)
    assert log_prob.size() == torch.Size([])


def test_incorrect_input_dimensions():
    x = torch.linspace(0, 1, 10).unsqueeze(1)
    y = torch.sin(x * (2 * 3.14))
    prior_mean_value = 0.0
    lengthscale = 0.1
    scale = 1.0

    mean_function = lambda x: prior_mean_value * torch.ones(x.size(0))
    kernel = gp.kernels.RBF(input_dim=1, variance=torch.tensor(scale), lengthscale=torch.tensor(lengthscale))
    gp_prior = GPPrior(mean_function=mean_function, kernel=kernel)

    with pytest.raises(ValueError):
        gp_prior.log_prob(x, y)


def test_inconsistent_xy_dimensions():
    x = torch.linspace(0, 1, 10)
    y = torch.randn(5)
    prior_mean_value = 0.0
    lengthscale = 0.1
    scale = 1.0

    mean_function = lambda x: prior_mean_value * torch.ones(x.size(0))
    kernel = gp.kernels.RBF(input_dim=1, variance=torch.tensor(scale), lengthscale=torch.tensor(lengthscale))
    gp_prior = GPPrior(mean_function=mean_function, kernel=kernel)

    with pytest.raises(ValueError):
        gp_prior.log_prob(x, y)
