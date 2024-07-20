from typing import Union, List, Tuple, Dict, Optional, Callable

import gymnasium as gym
import numpy as np
import torch

from ray.rllib.algorithms.dqn import DQNTorchPolicy
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import TensorStructType, TensorType


class TabularBoltzmannPolicy(Policy):
    """
    Creates a Boltzmann policy from a tabular Q_sa.
    """

    def __init__(self, Q_sa, observation_space, action_space, config=None, beta=1.0, onehot=True):
        config = config or {}
        Policy.__init__(self, observation_space, action_space, config)

        self.Q_sa = Q_sa
        assert isinstance(action_space, gym.spaces.Discrete), \
            "Action space must be discrete."
        # assert isinstance(observation_space, gym.spaces.Discrete), \
        #     "Observation space must be discrete."
        # assert Q_sa.shape == (observation_space.n, action_space.n), \
        #     "Q_sa must be of shape (observation_space.n, action_space.n)"

        self.onehot = onehot
        self.beta = beta

    def _get_action_distribution(self, obs):
        if self.onehot:
            obs = torch.argmax(torch.from_numpy(obs), keepdim=True)

        return torch.nn.functional.softmax(self.beta * self.Q_sa[obs, :], dim=-1)

    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs) -> Tuple[TensorType, List, Dict]:
        action_dist = self._get_action_distribution(torch.from_numpy(obs_batch))
        actions = torch.multinomial(action_dist, 1).squeeze(1)
        return actions, [], {}

    def compute_single_action(self,
                              obs: Optional[TensorStructType] = None,
                              state: Optional[List[TensorType]] = None,
                              **kwargs) -> Tuple[int, List, Dict]:
        action_dist = self._get_action_distribution(obs)
        action = torch.multinomial(action_dist, 1).squeeze()
        return action, [], {
            'action_prob': action_dist[action],
            'action_logp': torch.log(action_dist[action])
        }


import torch
import torch.nn as nn
import gymnasium as gym
from ray.rllib.policy.policy import Policy


class ContinuousBoltzmannPolicy(Policy):
    """
    Creates a Boltzmann policy using a neural network-based Q-function for continuous states and actions.
    """

    def __init__(self, model: nn.Module | Callable, observation_space: gym.Space, action_space: gym.Space,
                 config=None, beta=1.0, num_action_samples=10):
        config = config or {}
        Policy.__init__(self, observation_space, action_space, config)

        # The neural network-based Q-function model
        self.model = model
        self.beta = beta
        self.num_action_samples = num_action_samples

    def _get_action_distribution(self, obs_f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Sample multiple actions
        action_samples_af = torch.tensor(np.array([self.action_space.sample() for _ in range(self.num_action_samples)]),
                                      dtype=torch.float)
        repeated_obs_af = obs_f.repeat(self.num_action_samples, 1)

        # Get Q-values for each sampled action
        q_values = self.model(repeated_obs_af, action_samples_af)

        # Calculate probabilities using Boltzmann distribution
        action_dist = torch.nn.functional.softmax(self.beta * q_values, dim=0)

        return action_dist, action_samples_af, q_values

    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs) -> Tuple[torch.Tensor, List, Dict]:
        actions = []
        for obs in obs_batch:
            action_dist, sampled_actions, q_a = self._get_action_distribution(torch.from_numpy(obs))
            action = torch.multinomial(action_dist, 1).squeeze()
            actions.append(sampled_actions[action].numpy())

        return torch.tensor(actions), [], {}

    def compute_single_action(self,
                              obs: Optional[torch.Tensor] = None,
                              state: Optional[List[torch.Tensor]] = None,
                              **kwargs) -> Tuple[int, List, Dict]:
        action_dist, sampled_actions, q_a = self._get_action_distribution(torch.tensor(obs, dtype=torch.float))
        action = torch.multinomial(action_dist.squeeze(1), 1).squeeze()
        selected_action = sampled_actions[action].numpy()

        return selected_action, [], {
            'action_prob': action_dist[action].item(),
            'action_logp': torch.log(action_dist[action]).item(),
            'action_q_value': q_a[action].item(),
            'sampled_actions': sampled_actions,
            'action_dist': action_dist,
            'q_values': q_a
        }


def boltzmann_logprobs(q_vals_ba: TensorType, beta_expert: Union[float, TensorType]):
    # If q_vals_ba is a numpy array, cast to torch tensor
    if isinstance(q_vals_ba, np.ndarray):
        q_vals_ba = torch.from_numpy(q_vals_ba)
    log_probs_ba = beta_expert * q_vals_ba - torch.logsumexp(beta_expert * q_vals_ba, dim=-1, keepdim=True)
    return log_probs_ba


class BoltzmannWrapperPolicy(DQNTorchPolicy):
    """
    Creates a Boltzmann policy from a Ray DQN policy.
    """

    def __init__(self, dqn_policy: DQNTorchPolicy, demo_config: 'BoltzmannDemosFromRLCheckpointConfig'):
        super().__init__(obs_space=dqn_policy.observation_space, action_space=dqn_policy.action_space,
                         config=dqn_policy.config)
        self.dqn_policy = dqn_policy
        self.boltzmann_coeff = demo_config.beta_expert

    def compute_actions(
            self,
            obs_batch_bf: Union[List[TensorStructType], TensorStructType],
            state_batches=None,
            **kwargs,
    ) -> Tuple[TensorStructType, List[TensorType], Dict[str, TensorType]]:
        actions_orig, _, extras = self.dqn_policy.compute_actions(obs_batch_bf, state_batches, **kwargs)
        q_vals_ba = extras['q_values']
        new_logprobs_ba = boltzmann_logprobs(q_vals_ba, self.boltzmann_coeff)
        new_probs_ba = torch.exp(new_logprobs_ba)
        new_actions_b = torch.distributions.categorical.Categorical(new_probs_ba).sample()
        action_logprobs = new_logprobs_ba[torch.arange(len(new_probs_ba)), new_actions_b]
        action_probs_b = new_probs_ba[torch.arange(len(new_probs_ba)), new_actions_b]

        return new_actions_b, [], {
            'action_dist_inputs': extras['action_dist_inputs'],
            'action_prob': action_probs_b,
            'action_logp': action_logprobs
        }

    def compute_single_action(self,
                              obs: Optional[TensorStructType] = None,
                              state: Optional[List[TensorType]] = None,
                              **kwargs):
        action_orig, _, extras = self.dqn_policy.compute_single_action(obs, state, **kwargs)
        q_vals_a = extras['q_values']
        new_logprobs_a = boltzmann_logprobs(q_vals_a, self.boltzmann_coeff)
        new_probs_a = torch.exp(new_logprobs_a)
        new_action = torch.distributions.categorical.Categorical(new_probs_a).sample()
        new_action_logprob = new_logprobs_a[new_action]
        new_action_prob = new_probs_a[new_action]

        return new_action, [], {
            'action_dist_inputs': extras['action_dist_inputs'],
            'action_prob': new_action_prob,
            'action_logp': new_action_logprob
        }
