from typing import Callable

import torch
import pyro.contrib.gp as gp
import pyro.distributions as dist


class GPPrior:
    """
    Gaussian Process prior (primarily used for continuous-space reward functions).

    :param mean_function: Mean function of the GP prior.
    :param kernel: Kernel of the GP prior.
    """
    def __init__(self,
                 mean_function: Callable[[torch.Tensor], torch.Tensor] = None,
                 kernel: Callable[[torch.Tensor], torch.Tensor] = None):

        self.mean_function = mean_function
        if self.mean_function is None:
            self.mean_function = lambda x: torch.zeros(x.size(0), requires_grad=False)

        self.kernel = kernel
        if self.kernel is None:
            self.kernel = gp.kernels.RBF(input_dim=1)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor, jitter=1e-1) -> torch.Tensor:
        # Compute the covariance matrix
        K = self.kernel.forward(x) + torch.eye(x.size(0)) * jitter  # Adding a small jitter for numerical stability
        # Ensure symmetry (numerical imprecisions in kernel computation can lead to the PositiveDefinite constraint failing)
        K = (K + K.t()) / 2.
        # Define the multivariate normal distribution
        mvn = dist.MultivariateNormal(self.mean_function(x), K)

        # Compute log probability
        log_prob = mvn.log_prob(y)

        return log_prob


class PrecomputedGPPrior:
    """
    Gaussian Process prior (primarily used for continuous-space reward functions).

    :param mean_function: Mean function of the GP prior.
    :param kernel: Kernel of the GP prior.
    """
    def __init__(self,
                 eval_points: torch.Tensor = None,
                 mean_function: Callable[[torch.Tensor], torch.Tensor] = None,
                 kernel: Callable[[torch.Tensor], torch.Tensor] = None,
                 jitter=1e-3):

        self.mean_function = mean_function
        if self.mean_function is None:
            self.mean_function = lambda x: torch.zeros(x.size(0))

        self.kernel = kernel
        if self.kernel is None:
            self.kernel = gp.kernels.RBF(input_dim=1)

        self.dist = None

        if eval_points is not None:
            self.precompute(eval_points, jitter)

    def precompute(self, x_eval: torch.Tensor, jitter=1e-1):
        # Compute the covariance matrix
        K = self.kernel.forward(x_eval) + torch.eye(x_eval.size(0)) * jitter
        # Ensure symmetry (numerical imprecisions in kernel computation can break the PositiveDefinite constraint)
        K = (K + K.t()) / 2.
        # Define the multivariate normal distribution
        self.dist = dist.MultivariateNormal(self.mean_function(x_eval).detach(), K.detach())

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        # Compute log probability
        log_prob = self.dist.log_prob(y)

        return log_prob
