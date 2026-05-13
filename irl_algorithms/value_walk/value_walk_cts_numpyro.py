"""
This script aims to re-implement the ValueWalk algorithm using Numpyro.
The code is experimental and not yet extensively tested for correctness of inference.
    
"""

from __future__ import annotations
from typing import List, Callable, Optional, Tuple
import logging

import numpy as np
import gymnasium as gym
import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from numpyro.infer import NUTS, HMC
import torch
from torch.quasirandom import SobolEngine
from irl_algorithms.demonstrations import Demonstrations
from irl_algorithms.irl_method import IRLMethod
from models import lm_jax
from models.reward_models.q_based_reward_model import QBasedSampleBasedRewardModel
from irl_algorithms.mcmc_irl import BayesianIRLConfig

VW_Q_PARAM_KEY = 'theta_q'

if torch.cuda.is_available():
    TORCH_DEVICE = torch.device('cuda')
    idx = TORCH_DEVICE.index
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    JAX_DEVICE = jax.devices('gpu')[idx]
else:
    TORCH_DEVICE = torch.device('cpu')
    JAX_DEVICE = jax.devices('cpu')[0]


def _rng_key() -> jnp.ndarray:
    return jax.device_put(
        PRNGKey(int(0)),
        device=JAX_DEVICE,
    )

# JAX-Torch conversion helpers
def _torch_to_jax(tensor: torch.Tensor, device=JAX_DEVICE, dtype: Any=jnp.float32) -> jnp.ndarray:
        return jax.device_put(jnp.array(tensor.detach().cpu().numpy(), dtype=dtype), device=device)

def _jax_to_torch(array: jnp.ndarray, device=TORCH_DEVICE):
        return torch.from_numpy(np.array(array)).to(device)

def value_walk_model_approx_cts_numpyro(x_daf: jnp.ndarray, theta_q_prior: jnp.ndarray, beta_expert: jnp.ndarray, a_df: jnp.ndarray | None=None,
                                bayesian_module=lm_jax):

    theta_q = numpyro.sample(VW_Q_PARAM_KEY, theta_q_prior)

    q_da = bayesian_module(x_daf, theta_q).squeeze(axis=-1)

    likelihood_dist = dist.Categorical(logits=beta_expert*q_da)

    with numpyro.plate('data', x_daf.shape[0]):
        return numpyro.sample('obs', likelihood_dist, obs=jnp.argmax(a_df, axis=1) if a_df is not None else None)

class QParamPriorCtsNumpyro(dist.Distribution):
    """
    Prior distribution over Q-values implied by the prior over rewards as used by the continuous version
    of the ValueWalk algorithm. This largely mirrors the torch implementation, but replaces the internal
    distribution and callable objects with JAX/Numpyro objects, focussing on the sample() and log_prob() functions.
    """

    support = dist.constraints.real_vector

    def __init__(self,
                 r_prior,
                 action_set_af: torch.Tensor,
                 preprocessing_module: torch.nn.Module,
                 bayesian_module: Callable,
                 env_sim: gym.Env,
                 num_params: int,
                 evaluation_trajectories: Demonstrations = None,
                 approximate_sampling_dist: dist.Distribution = None,
                 use_cheating_sample: bool = True,
                 gamma: float = 0.9,
                 final_q_to_r: bool = False):
        """
        :param r_prior: Reward prior
        :param preprocessing_module: Preprocessing module
        :param bayesian_module: A function from an input tensor of features and a param tensor to a q_value
        :param env_sim: Environment simulator (gym env)
        :param approximate_sampling_dist: A distribution over the parameters of the bayesian module used only
                to get the initial MCMC sample
        :param device:
        :param use_cheating_sample: If True, the sample is obtained by sampling from the approximate_sampling_dist
            (this is to alert to the fact that the sample method does not actually sample from the prior)
        :param gamma: Discount factor
        :param num_action_samples: Number of action samples used to compute the q_values
        """
        super().__init__(batch_shape=(), event_shape=(num_params,))

        self.r_prior_c = r_prior
        self.candidate_actions = action_set_af
        self.preprocessing_module = preprocessing_module
        self.bayesian_module = bayesian_module
        self.env_sim = env_sim
        self.evaluation_trajectories = evaluation_trajectories

        self.gamma = gamma
        self.final_q_to_r = final_q_to_r

        self.key = _rng_key()

        self.param_shape = (num_params,)

        if approximate_sampling_dist is None:
            self.approximate_sampling_dist = dist.Normal(jnp.zeros(self.param_shape, dtype=float),
                                                         jnp.ones(self.param_shape, dtype=float)).to_event(1)
        else:
            self.approximate_sampling_dist = approximate_sampling_dist
        self._use_cheating_sample = use_cheating_sample

        self.x_eval_bf, self.x_eval_next_baf = self.prepare_eval_arrays()

        logging.info("QParamPriorCts initialized with the following (discretized) candidate actions: %s", self.candidate_actions)

    def prepare_eval_arrays(self):
        """
        Prepare tensors for evaluating the q_values of the evaluation trajectories.
        """
        x_list = []
        x_next_list = []
        for traj in self.evaluation_trajectories:
            traj_oa_tensor = traj.get_oa_tensor().float().to(TORCH_DEVICE)
            # Get the dimensions of the states tensor and the candidate actions tensor
            state_feats = traj.states_tensor.shape[-1]
            action_feats = self.candidate_actions.shape[-1]

            # Broadcast the tensors to be ready for concatenation:
            states_next_baf = traj.states_tensor[1:, None, :].expand(-1, self.candidate_actions.shape[0], state_feats).float().to(TORCH_DEVICE)
            actions_baf = self.candidate_actions[None, :, :].expand(states_next_baf.shape[0], -1, action_feats)

            x_next_baf = torch.cat([states_next_baf, actions_baf], dim=-1)

            if self.final_q_to_r:
                x_list.append(traj_oa_tensor)
                x_next_list.append(torch.cat([x_next_baf, torch.zeros((1,)+x_next_baf.shape[1:], dtype=torch.float)],
                                             dim=0))
            else:
                x_list.append(traj_oa_tensor[:-1, :])
                x_next_list.append(x_next_baf)

        x_eval_bf = torch.cat(x_list, dim=0)
        x_eval_next_baf = torch.cat(x_next_list, dim=0).detach()

        if hasattr(self.r_prior_c, "precompute"):
            # Precompute the reward prior for the evaluation trajectories (typically precomputes the covariance matrix)
            self.r_prior_c.precompute(_torch_to_jax(x_eval_bf))

        if self.preprocessing_module is not None:
            x_eval_bf = self.preprocessing_module(x_eval_bf)
            x_eval_next_baf = self.preprocessing_module(x_eval_next_baf)

        return _torch_to_jax(x_eval_bf), _torch_to_jax(x_eval_next_baf.to(TORCH_DEVICE).clone(memory_format=torch.contiguous_format))

    def sample(self):
        """
        WARNING: this is not a sample from the prior, but from the approximate sampling distribution
        It's supposed to just be used to get the initial MCMC sample.
        :return:
        """
        if self._use_cheating_sample:
            q_params = self.approximate_sampling_dist.sample(self.key)
        else:
            raise NotImplementedError
        return q_params

    def log_prob(self, q_params: jnp.ndarray) -> jnp.ndarray:

        r_b = self.calculate_implied_rewards(q_params)

        q_logprior = self.r_prior_c.log_prob(self.x_eval_bf, r_b)
        # q_logprior = self.r_prior_c.log_prob(r_b)


        return q_logprior

    def calculate_implied_rewards(self, q_params: jnp.ndarray
                                  ) -> List[jnp.ndarray]:
        current_q_b = self.bayesian_module(self.x_eval_bf, q_params).squeeze(-1)
        next_q_ba = self.bayesian_module(self.x_eval_next_baf, q_params).squeeze(-1)

        if self.final_q_to_r:
            next_q_ba[-1, :] = 0

        next_v_b = jnp.max(next_q_ba, axis=-1)[0]

        r_b = current_q_b - self.gamma * next_v_b

        return r_b

class ValueWalkCtsNumpyro(IRLMethod):

    def __init__(self, env: gym.Env, config: BayesianIRLConfig):
        super().__init__(env, config)
        self.reward_prior = config.reward_prior_factory()
        self.config = config # not needed techincally, I am just annoyed at all the red lines downstream

        # Prepare a set of alternative actions
        self.action_set_af = self.prepare_actions(config.num_action_samples)

    def prepare_actions(self, num_points: int = None):
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            assert num_points is None
            return torch.eye(self.env.action_space.n, device=TORCH_DEVICE, dtype=torch.float)
        else:
            return self.initialize_sobol_actions(num_points)

    def initialize_sobol_actions(self, num_points: int):
        """
        Initialize a Sobol sequence of actions within the bounds of the action space.
        """
        sobol_engine = SobolEngine(dimension=self.env.action_space.shape[0], scramble=True, seed=7)
        # Generate points in [0, 1] range
        raw_points = sobol_engine.draw(num_points).to(TORCH_DEVICE)
        # Scale points to the range of each dimension of the action space
        lower_bounds = torch.tensor(self.env.action_space.low, dtype=torch.float, device=TORCH_DEVICE)
        upper_bounds = torch.tensor(self.env.action_space.high, dtype=torch.float, device=TORCH_DEVICE)
        scaled_points = lower_bounds + (upper_bounds - lower_bounds) * raw_points
        return scaled_points.to(TORCH_DEVICE)

    def prepare_feature_vector(self, s_df, a_df, preprocessing_module):

        actions_daf = torch.broadcast_to(self.action_set_af[None, :, :],
                                         s_df.shape[:-1] + self.action_set_af.shape)

        if not isinstance(self.env.action_space, gym.spaces.Discrete):
            actions_daf = torch.cat([a_df[:, None, :], actions_daf], dim=-2)
        s_daf = torch.broadcast_to(s_df[:, None, :], actions_daf.shape[:-1] + s_df.shape[-1:])
        x_daf = torch.cat([s_daf, actions_daf], dim=-1)

        if preprocessing_module is not None:
            x_daf = preprocessing_module(x_daf)

        return x_daf

    def run_mcmc(self, s_df: torch.Tensor, a_df: torch.Tensor, theta_q_prior, preprocessing_module: Optional[torch.nn.Module] = None):
        print("Starting the MCMC phase")

        x_daf = self.prepare_feature_vector(s_df, a_df, preprocessing_module=preprocessing_module)

        x_daf_jnp = _torch_to_jax(x_daf)
        a_df_jnp = _torch_to_jax(a_df)

        if self.config.hmc_use_nuts:
            mcmc_kernel = NUTS(value_walk_model_approx_cts_numpyro,
                                        step_size=self.config.hmc_step_size,
                                        adapt_step_size=self.config.hmc_adapt_step_size,
                                        adapt_mass_matrix=self.config.hmc_adapt_mass_matrix,
                                        target_accept_prob=self.config.hmc_target_accept_prob,)
                                        #full_mass=self.config.hmc_full_mass)
        else:
            mcmc_kernel = HMC(value_walk_model_approx_cts_numpyro,
                                        num_steps=self.config.hmc_num_steps,
                                        step_size=self.config.hmc_step_size,
                                        adapt_step_size=self.config.hmc_adapt_step_size,
                                        adapt_mass_matrix=self.config.hmc_adapt_mass_matrix,
                                        target_accept_prob=self.config.hmc_target_accept_prob,)
                                        #full_mass=self.config.hmc_full_mass)


        mcmc = numpyro.infer.MCMC(mcmc_kernel,
                               num_samples=self.config.num_samples,
                               num_warmup=self.config.warmup_steps,
                               num_chains=self.config.num_chains,
                               chain_method = 'vectorized')
        
        rng_key = _rng_key()
        
        mcmc.run(rng_key = rng_key,
                 x_daf=x_daf_jnp,
                 a_df=a_df_jnp,
                 theta_q_prior=theta_q_prior,
                 beta_expert=self.config.beta_expert,
                 bayesian_module=self.config.q_model)

        samples = mcmc.get_samples(group_by_chain=False)

        return samples


    def run(self, demonstrations_t: Demonstrations, preprocessing_module=None,
            ) -> Tuple[QBasedSampleBasedRewardModel, Optional[torch.nn.Module]]:

        a_df = demonstrations_t.actions_tensor.float().to(TORCH_DEVICE)
        s_df = demonstrations_t.states_tensor.float().to(TORCH_DEVICE)

        if self.config.preprocessing_module_factory is not None and preprocessing_module is None:
            preprocessing_module = self.config.preprocessing_module_factory()

        if self.config.aux_demo_factory is not None:
            aux_demos = self.config.aux_demo_factory(D=demonstrations_t, env=self.env)
        else:
            aux_demos = demonstrations_t

        theta_q_prior = QParamPriorCtsNumpyro(
            self.reward_prior,
            preprocessing_module=preprocessing_module,
            bayesian_module=self.config.q_model,
            env_sim=self.env,
            num_params=self.config.q_model_params,
            use_cheating_sample=True,
            gamma=self.config.gamma,
            evaluation_trajectories=aux_demos,
            action_set_af=self.action_set_af,
            final_q_to_r=self.config.final_q_to_r,
            )

        samples = self.run_mcmc(s_df=s_df, a_df=a_df, theta_q_prior=theta_q_prior,
                                preprocessing_module=preprocessing_module)

        info = {}

        print(samples['theta_q'].shape, type(samples['theta_q']))
        samples['theta_q'] = _jax_to_torch(samples['theta_q'])
        print(samples['theta_q'].shape, type(samples['theta_q']))

        return QBasedSampleBasedRewardModel(q_param_samples=samples,
                                            q_model=self.config.q_model,
                                            preprocessing_module=preprocessing_module), info
