from typing import Callable

import pyro.distributions
import torch
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel

from irl_algorithms.irl_method import IRLConfig


class BayesianIRLConfig(IRLConfig):

    reward_prior_factory: Callable[['IRLConfig'], pyro.distributions.Distribution]
    prior_mean_factory: Callable[[], Callable[[torch.Tensor], torch.Tensor]] = None
    prior_kernel_factory: Callable[[], Callable[[torch.Tensor], torch.Tensor]] = None
    prior_mean: float = 0.
    prior_scale: float = 1.0

    reward_eval_points: int = None

    kernel: Callable = None

    num_samples: int = 1000
    warmup_steps: int = 1000
    num_chains: int = 1

    hmc_step_size: float = 0.1
    hmc_full_mass: bool = True
    hmc_num_steps: int = 10
    hmc_adapt_step_size: bool = True
    hmc_adapt_mass_matrix: bool = True
    hmc_target_accept_prob: float = 0.8
    hmc_use_nuts: bool = True
    pyro_jit_compile: bool = True

    svi_iters: int = 1000
    svi_lr: float = 0.01
    svi_reporting_frequency: int = 1000

    constraint_weight: float = 1.0  # lambda in AVRIL
    state_only: bool = False  # whether to use state-only reward model in AVRIL
    final_q_to_r: bool = False  # whether to use final q value as reward


def get_pyro_mcmc_kernel(model: Callable, config: BayesianIRLConfig) -> MCMCKernel:
    
    if config.hmc_use_nuts:
        mcmc_kernel = pyro.infer.NUTS(model,
                                      step_size=config.hmc_step_size,
                                      adapt_step_size=config.hmc_adapt_step_size,
                                      adapt_mass_matrix=config.hmc_adapt_mass_matrix,
                                      target_accept_prob=config.hmc_target_accept_prob,
                                      full_mass=config.hmc_full_mass,
                                      jit_compile=config.pyro_jit_compile)
    else:
        mcmc_kernel = pyro.infer.HMC(model,
                                     num_steps=config.hmc_num_steps,
                                     step_size=config.hmc_step_size,
                                     adapt_step_size=config.hmc_adapt_step_size,
                                     adapt_mass_matrix=config.hmc_adapt_mass_matrix,
                                     target_accept_prob=config.hmc_target_accept_prob,
                                     full_mass=config.hmc_full_mass,
                                     jit_compile=config.pyro_jit_compile)
        
    return mcmc_kernel
