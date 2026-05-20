import torch
import pyro.distributions as dist

from configuration.configurable_factory import configurable_factory
from models.gp.gp_prior import GPPrior, PrecomputedGPPrior
from models.gp.gp_prior_jax import GPPriorJax, PrecomputedGPPriorJax


@configurable_factory
def get_independent_uniform_prior(config: 'MCMCIRLConfig'):

    def prior_factory():
        low = -100.0
        high = 0.0
        r_prior = dist.Independent(
            dist.Uniform(
                low=torch.full((config.reward_eval_points,), low, dtype=torch.float),
                high=torch.full((config.reward_eval_points,), high, dtype=torch.float)
            ),
            1
        )
        return r_prior

    return prior_factory




@configurable_factory
def get_independent_normal_prior(config: 'MCMCIRLConfig'):

    def prior_factory():
        r_prior_var = config.prior_scale ** 2
        m_r_prior_s = torch.ones(torch.Size([config.reward_eval_points]), dtype=torch.float) * config.prior_mean
        K_r_prior_ss = r_prior_var * torch.torch.eye(config.reward_eval_points, dtype=torch.float)
        r_prior = dist.MultivariateNormal(
            loc=m_r_prior_s,
            covariance_matrix=K_r_prior_ss
        )
        return r_prior

    return prior_factory


@configurable_factory
def get_gp_prior(config: 'MCMCIRLConfig'):

    def prior_factory():
        # Assuming the config object has attributes for GP prior configurations
        # like prior_mean_factory and kernel_factory which return the respective mean function and kernel.

        mean_function = config.prior_mean_factory()
        kernel = config.prior_kernel_factory()

        # Create an instance of the GPPrior
        gp_prior_instance = GPPrior(mean_function=mean_function, kernel=kernel)

        return gp_prior_instance

    return prior_factory


@configurable_factory
def get_static_evals_gp_prior(config: 'MCMCIRLConfig'):

    def prior_factory():
        # Assuming the config object has attributes for GP prior configurations
        # like prior_mean_factory and kernel_factory which return the respective mean function and kernel.

        mean_function = config.prior_mean_factory()
        kernel = config.prior_kernel_factory()

        # Create an instance of the GPPrior
        gp_prior_instance = PrecomputedGPPrior(mean_function=mean_function, kernel=kernel)

        return gp_prior_instance

    return prior_factory


@configurable_factory
def get_gp_prior_jax(config: 'MCMCIRLConfig'):

    def prior_factory():
        # Assuming the config object has attributes for GP prior configurations
        # like prior_mean_factory and kernel_factory which return the respective mean function and kernel.

        mean_function = config.prior_mean_factory()
        kernel = config.prior_kernel_factory()

        # Create an instance of the GPPrior
        gp_prior_jax = GPPriorJax(mean_function=mean_function, kernel=kernel)

        return gp_prior_jax

    return prior_factory

@configurable_factory
def get_static_evals_gp_prior_jax(config: 'MCMCIRLConfig'):

    def prior_factory():

        mean_function = config.prior_mean_factory()
        kernel = config.prior_kernel_factory()

        gp_prior_jax = PrecomputedGPPriorJax(mean_function=mean_function, kernel=kernel)
        
        return gp_prior_jax
    
    return prior_factory