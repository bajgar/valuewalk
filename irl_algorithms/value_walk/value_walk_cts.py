from typing import List, Tuple, Callable, Dict, Optional, Union
import logging

import gymnasium as gym
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from torch.quasirandom import SobolEngine

from irl_algorithms.demonstrations import Demonstrations, Trajectory
from irl_algorithms.irl_method import IRLMethod
from irl_algorithms.mcmc_irl import BayesianIRLConfig, get_pyro_mcmc_kernel
from models import linear_model
from models.reward_models.linear_reward_model import MonteCarloLinearRewardModel
from models.reward_models.q_based_reward_model import QBasedSampleBasedRewardModel
from rl_algorithms.policies.boltzmann import ContinuousBoltzmannPolicy


if torch.cuda.is_available():
    default_device = torch.device('cuda')
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    default_device = torch.device('cpu')

VW_Q_PARAM_KEY = 'theta_q'


class QParamPriorCts(dist.Distribution):
    """
    Prior distribution over Q-values implied by the prior over rewards as used by the continuous version
    of the ValueWalk algorithm.
    """

    support = dist.constraints.real_vector

    def __init__(self,
                 r_prior: torch.distributions.Distribution,
                 action_set_af: torch.Tensor,
                 preprocessing_module: torch.nn.Module,
                 bayesian_module: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 env_sim: gym.Env,
                 num_params: int,
                 evaluation_trajectories: Demonstrations = None,
                 approximate_sampling_dist: dist.Distribution = None,
                 device=default_device,
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

        self.r_prior_c = r_prior
        self.candidate_actions = action_set_af
        self.preprocessing_module = preprocessing_module
        self.bayesian_module = bayesian_module
        self.env_sim = env_sim
        self.evaluation_trajectories = evaluation_trajectories

        self.gamma = gamma
        self.final_q_to_r = final_q_to_r

        self.device = device

        self.param_shape = (num_params,)

        if approximate_sampling_dist is None:
            self.approximate_sampling_dist = dist.Normal(torch.zeros(self.param_shape, dtype=torch.float),
                                                         torch.ones(self.param_shape, dtype=torch.float)).to_event(1)
        else:
            self.approximate_sampling_dist = approximate_sampling_dist
        self._use_cheating_sample = use_cheating_sample

        self.x_eval_bf, self.x_eval_next_baf = self.prepare_eval_tensors()

        logging.info("QParamPriorCts initialized with the following (discretized) candidate actions: ", self.candidate_actions)

    def prepare_eval_tensors(self):
        """
        Prepare tensors for evaluating the q_values of the evaluation trajectories.
        """
        x_list = []
        x_next_list = []
        for traj in self.evaluation_trajectories:
            traj_oa_tensor = traj.get_oa_tensor().float().to(self.device)
            # Get the dimensions of the states tensor and the candidate actions tensor
            state_feats = traj.states_tensor.shape[-1]
            action_feats = self.candidate_actions.shape[-1]

            # Broadcast the tensors to be ready for concatenation:
            states_next_baf = traj.states_tensor[1:, None, :].expand(-1, self.candidate_actions.shape[0], state_feats).float().to(self.device)
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
            self.r_prior_c.precompute(x_eval_bf)

        if self.preprocessing_module is not None:
            x_eval_bf = self.preprocessing_module(x_eval_bf)
            x_eval_next_baf = self.preprocessing_module(x_eval_next_baf)

        return x_eval_bf.detach(), x_eval_next_baf.detach().to(self.device).clone(memory_format=torch.contiguous_format)

    def sample(self):
        """
        WARNING: this is not a sample from the prior, but from the approximate sampling distribution
        It's supposed to just be used to get the initial MCMC sample.
        :return:
        """
        if self._use_cheating_sample:
            q_params = self.approximate_sampling_dist.sample()
        else:
            raise NotImplementedError
        return q_params

    def log_prob(self, q_params: torch.Tensor) -> torch.Tensor:

        r_b = self.calculate_implied_rewards(q_params)

        q_logprior = self.r_prior_c.log_prob(self.x_eval_bf, r_b)
        # q_logprior = self.r_prior_c.log_prob(r_b)


        return q_logprior

    def calculate_implied_rewards(self, q_params: torch.Tensor
                                  ) -> List[torch.Tensor]:
        current_q_b = self.bayesian_module(self.x_eval_bf, q_params).squeeze(-1)
        next_q_ba = self.bayesian_module(self.x_eval_next_baf, q_params).squeeze(-1)

        if self.final_q_to_r:
            next_q_ba[-1, :] = 0

        next_v_b = next_q_ba.max(dim=-1)[0]

        r_b = current_q_b - self.gamma * next_v_b

        return r_b


def value_walk_model_approx_cts(x_daf, theta_q_prior, beta_expert, a_df=None,
                                bayesian_module=linear_model):

    theta_q = pyro.sample(VW_Q_PARAM_KEY, theta_q_prior)

    q_da = bayesian_module(x_daf, theta_q).squeeze(-1)

    likelihood_dist = dist.Categorical(logits=beta_expert*q_da)

    with pyro.plate('data', x_daf.shape[0]):
        return pyro.sample('obs', likelihood_dist, obs=torch.argmax(a_df, dim=1) if a_df is not None else None)


class ValueWalkCts(IRLMethod):

    def __init__(self, env: gym.Env, config: BayesianIRLConfig, device=default_device):
        super().__init__(env, config)
        self.reward_prior = config.reward_prior_factory()
        self.device = device

        # Prepare a set of alternative actions
        self.action_set_af = self.prepare_actions(config.num_action_samples)

    def prepare_actions(self, num_points: int = None):
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            assert num_points is None
            return torch.eye(self.env.action_space.n, device=self.device, dtype=torch.float)
        else:
            return self.initialize_sobol_actions(num_points)

    def initialize_sobol_actions(self, num_points: int):
        """
        Initialize a Sobol sequence of actions within the bounds of the action space.
        """
        sobol_engine = SobolEngine(dimension=self.env.action_space.shape[0], scramble=True, seed=7)
        # Generate points in [0, 1] range
        raw_points = sobol_engine.draw(num_points).to(self.device)
        # Scale points to the range of each dimension of the action space
        lower_bounds = torch.tensor(self.env.action_space.low, dtype=torch.float, device=self.device)
        upper_bounds = torch.tensor(self.env.action_space.high, dtype=torch.float, device=self.device)
        scaled_points = lower_bounds + (upper_bounds - lower_bounds) * raw_points
        return scaled_points.to(self.device)

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

        pyro.clear_param_store()

        x_daf = self.prepare_feature_vector(s_df, a_df, preprocessing_module=preprocessing_module)

        mcmc_kernel = get_pyro_mcmc_kernel(value_walk_model_approx_cts, self.config)

        def checkpoint_hook(kernel, samples, stage, i):
            if self.config.checkpoint_frequency is not None and i % self.config.checkpoint_frequency == 0:
                torch.save({
                    'info': {
                        # 'config': self.config.json(),
                        'stage': stage,
                        'step': i},
                    'samples': samples
                }, self.config.checkpoint_path)
                print(f"Checkpoint saved at iteration {i} into {self.config.checkpoint_path}")

        if self.config.num_chains == 1 or not self.config.num_chains:
            init_params = {VW_Q_PARAM_KEY: theta_q_prior.sample()}
        else:
            init_params = {VW_Q_PARAM_KEY: torch.stack([theta_q_prior.sample() for _ in range(self.config.num_chains)])}
        mcmc = pyro.infer.MCMC(mcmc_kernel,
                               num_samples=self.config.num_samples,
                               warmup_steps=self.config.warmup_steps,
                               num_chains=self.config.num_chains,
                               initial_params=init_params,
                               disable_validation=False,
                               hook_fn=checkpoint_hook if self.config.checkpoint_frequency is not None else None)

        mcmc.run(x_daf=x_daf.detach().clone(memory_format=torch.contiguous_format),
                 a_df=a_df,
                 theta_q_prior=theta_q_prior,
                 beta_expert=self.config.beta_expert,
                 bayesian_module=self.config.q_model)

        samples = mcmc.get_samples(group_by_chain=False)

        return samples


    def run(self, demonstrations_t: Demonstrations, preprocessing_module=None,
            ) -> (MonteCarloLinearRewardModel, Optional[torch.nn.Module]):

        a_df = demonstrations_t.actions_tensor.float().to(self.device)
        s_df = demonstrations_t.states_tensor.float().to(self.device)

        if self.config.preprocessing_module_factory is not None and preprocessing_module is None:
            preprocessing_module = self.config.preprocessing_module_factory()

        if self.config.aux_demo_factory is not None:
            aux_demos = self.config.aux_demo_factory(D=demonstrations_t, env=self.env)
        else:
            aux_demos = demonstrations_t

        theta_q_prior = QParamPriorCts(
            self.reward_prior,
            preprocessing_module=preprocessing_module,
            bayesian_module=self.config.q_model,
            env_sim=self.env,
            num_params=self.config.q_model_params,
            device=self.device,
            use_cheating_sample=True,
            gamma=self.config.gamma,
            evaluation_trajectories=aux_demos,
            action_set_af=self.action_set_af,
            final_q_to_r=self.config.final_q_to_r,
            )

        samples = self.run_mcmc(s_df=s_df, a_df=a_df, theta_q_prior=theta_q_prior,
                                preprocessing_module=preprocessing_module)

        info = {}

        return QBasedSampleBasedRewardModel(q_param_samples=samples,
                                            q_model=self.config.q_model,
                                            preprocessing_module=preprocessing_module), info
