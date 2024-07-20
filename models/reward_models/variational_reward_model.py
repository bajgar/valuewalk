from typing import Optional

import torch

from models.reward_models.reward_model import RewardModel


class NormalVariationalRewardModel(RewardModel):

    def __init__(self, reward_model: torch.nn.Module, q_model: torch.nn.Module = None):
        """
        Creates reward model based on a normal variational distribution over rewards. The variational distribution's
        parameters are parameterized by a neural network, which behaves depending on whether the action is discrete or
        continuous.
        - If the action is discrete, the neural network outputs a mean and log-variance for each action. (so the output
        dimension is 2 * action_dim)
        - If the action is continuous, the neural network outputs a mean and log variance for the reward.
        """
        self.reward_model = reward_model
        self.q_model = q_model

    def point_estimate(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Returns the point estimate of the reward.
        """
        return self.mean(s, a)

    def mean(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Returns the mean of the reward distribution.
        """
        if torch.is_floating_point(a):
            reward_params = self.reward_model(torch.cat([s, a], dim=-1))
            assert reward_params.shape[-1] == 2
            return reward_params[..., 0]

        else:
            reward_params = self.reward_model(s)
            means = reward_params[..., ::2]
            return means.gather(-1, a).squeeze(-1)

    def std(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Returns the standard deviation of the reward distribution.
        """
        return torch.sqrt(self.variance(s, a))

    def variance(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Returns the variance of the reward distribution.
        """
        if torch.is_floating_point(a):
            reward_params = self.reward_model(torch.cat([s, a], dim=-1))
            assert reward_params.shape[-1] == 2
            log_vars_b = reward_params[..., 1]

        else:
            reward_params = self.reward_model(s)
            log_vars_ba = reward_params[..., 1::2]
            log_vars_b = log_vars_ba.gather(-1, a).squeeze(-1)

        return torch.exp(log_vars_b)

    def pdf(self, s: torch.Tensor, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Returns the probability density of the reward distribution.
        """
        dist = torch.distributions.Normal(loc=self.mean(s, a), scale=self.std(s, a))
        return dist.log_prob(r)

    def quantile(self, s: torch.Tensor, a: torch.Tensor, q: float) -> torch.Tensor:
        """
        Returns the q-th quantile of the reward distribution.
        """
        dist = torch.distributions.Normal(loc=self.mean(s, a), scale=self.std(s, a))
        return dist.icdf(torch.tensor(q))

    def boltzmann_action_probs(self, s_bf: torch.Tensor, a_af_or_baf: torch.Tensor, beta: float,
                               density_over_volume: Optional[float] = None, discrete_actions=False,
                               return_logprobs=False) -> torch.Tensor:
        """
        Returns the action probabilities according to the Boltzmann distribution.
        :param s_bf: A batch of states
        :param a_af_or_baf: A batch of actions. This can either be the same batch of actions for each state (when
         actions have 2 dimansions - af) or a different batch of actions for each state (when actions have 3
            dimensions - baf).
        :param beta: Boltzmann rationality coefficient
        :param density_over_volume: If False (default), returns action probabilities. If True, returns action
            densities (treating actions as a Monte Carlo samples) where this parameter should a total volume by
            which to divide the action density.
        """
        if len(a_af_or_baf.shape) == len(s_bf.shape) == 2:
            s_baf = torch.broadcast_to(s_bf[:, None, :], [s_bf.shape[0], a_af_or_baf.shape[0], s_bf.shape[1]])
            a_baf = torch.broadcast_to(a_af_or_baf[None, :, :], [s_bf.shape[0], a_af_or_baf.shape[0], a_af_or_baf.shape[1]])
        elif len(s_bf.shape) == 2 and len(a_af_or_baf.shape) == 3:
            s_baf, a_baf = torch.broadcast_tensors(s_bf[:, None, :], a_af_or_baf)
        else:
            raise ValueError(f"Invalid shapes: {s_bf.shape} and {a_af_or_baf.shape}")
        x_baf = torch.cat([s_baf, a_baf], dim=-1)
        if discrete_actions:
            q_vals_ba = self.q_model(s_bf).squeeze(-1)
        else:
            q_vals_ba = self.q_model(x_baf).squeeze(-1)

        if return_logprobs:
            return torch.log_softmax(q_vals_ba * beta, dim=-1)

        action_probs_ba = torch.nn.functional.softmax(q_vals_ba * beta, dim=-1)
        if density_over_volume:
            action_probs_ba = action_probs_ba * action_probs_ba.shape[-1] / density_over_volume
        return action_probs_ba
