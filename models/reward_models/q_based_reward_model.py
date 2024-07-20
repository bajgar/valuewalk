from typing import Optional, Tuple, Dict, Callable

import botorch
import torch
import gymnasium as gym
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from irl_algorithms.demonstrations import Trajectory, Demonstrations
from irl_algorithms.utils.boltzmann import belief_weighed_action_probs
from models.reward_models.reward_model import RewardModel


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class QBasedSampleBasedRewardModel(RewardModel):
    def __init__(self, q_param_samples: Dict,
                 q_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 preprocessing_module: Optional[torch.nn.Module] = None,
                 reward_eval_states: Optional[torch.Tensor] = None,
                 reward_eval_actions: Optional[torch.Tensor] = None,
                 reward_samples: Optional[torch.Tensor] = None):
        """
        Creates a reward model based on a Q-function. The Q-function is assumed to be a neural network-based
        function that takes in a state-action pair and outputs a Q-value. The Q-function is assumed to be
        parameterized by a set of parameters. The reward model is based on a set of samples of the Q-function
        parameters. The reward model can be used to estimate the reward for a given state-action pair by
        using the Bellman equation.
        :param q_param_samples:
        :param q_model:
        :param preprocessing_module:
        :param reward_eval_states:
        :param reward_eval_actions:
        :param reward_samples:
        """
        # self.q_model = q_model
        self.q_param_samples = q_param_samples
        self.preprocessing_module = preprocessing_module

        assert reward_samples is None or (reward_eval_states is not None and reward_eval_actions is not None), \
            "reward_samples, reward_eval_states and reward_eval_actions must be provided together."
        self.reward_eval_states = reward_eval_states
        self.reward_eval_actions = reward_eval_actions
        self.reward_samples = reward_samples

        self.reward_means = None
        self.reward_vars = None

    def point_estimate(self, s_bf: torch.Tensor, a_bf: torch.Tensor) -> torch.Tensor:
        q_bc = self.q_samples(s_bf, a_bf)
        return torch.mean(q_bc, dim=1)

    def q_samples(self, s_bf: torch.Tensor, a_baf: torch.Tensor) -> torch.Tensor:
        """
        Provides samples from the Q function for the given state-action pairs corresponding to the param samples
        stored by the model.
        :param s_bf: A batch of states
        :param a_baf: A batch of actions. This can either be a single action for each state (when the batch
            dimensions match) or a batch of actions for each state (when the batch dimensions don't match).
        :return: q_bc: A batch of Q-values for each state-action pair or q_bac: a batch of Q-values for each
            each state and each action.
        """

        if len(s_bf) != len(a_baf):
            assert len(s_bf.shape) == len(a_baf.shape) == 2

            # Broadcast the state and action batches to have the same number of dimensions
            s_bf = torch.broadcast_to(s_bf[:, None, :], [s_bf.shape[0], a_baf.shape[0], s_bf.shape[1]])
            a_baf = torch.broadcast_to(a_baf[None, :, :], [s_bf.shape[0], a_baf.shape[0], a_baf.shape[1]])

            # Concatenate the state and action batches
            x_bf = torch.cat([s_bf, a_baf], dim=-1)

        else:
            if len(a_baf.shape) == 3:
                s_bf = torch.broadcast_to(s_bf[:, None, :], [s_bf.shape[0], a_baf.shape[1], s_bf.shape[1]])
            x_bf = torch.cat([s_bf, a_baf], dim=-1)

        if self.preprocessing_module is not None:
            x_bf = self.preprocessing_module(x_bf)

        params = self.q_param_samples['theta_q']

        if len(params.shape) == 3:
            params = params.reshape(params.shape[0] * params.shape[1], params.shape[2])

        q_bc = self.q_model(x_bf, params).squeeze(-1)

        return q_bc

    def traj_rewards(self, trajectory: Trajectory, gamma: float = 0.9, a_af: Optional[torch.Tensor] = None,
                     ignore_last: bool = True) -> torch.Tensor:
        """
        Computes the rewards for each state-action pair in the trajectory using the Q function.
        :param trajectory:
        :param gamma:
        :param a_af:
        :param ignore_last: whether to ignore the last state-action pair in the trajectory. If not ignored,
            the reward for the last state-action pair will be set to the Q-value.
        :return:
        """
        s_tf = trajectory.states_tensor
        a_tf = trajectory.actions_tensor

        q_tc = self.q_samples(s_tf, a_tf)

        q_current_tc = q_tc[:-1, :]

        if a_af is not None:
            q_next_tac = self.q_samples(s_tf[1:, :], a_af)
            v_next_tc = torch.max(q_next_tac, dim=-2)[0]
        else:
            v_next_tc = q_tc[1:, :]

        r_bc = q_current_tc - gamma * v_next_tc

        if not ignore_last:
            r_bc = torch.cat([r_bc, q_tc[-1:, :]], dim=0)

        return r_bc

    def calculate_reward_samples(self, demos: Demonstrations, gamma: float = 0.9, a_af: Optional[torch.Tensor] = None,
                                 ignore_episode_end: bool = True
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the rewards for each state-action pair in the trajectory using the Q function.
        :param demos: a Demonstrations object containing trajectories over which rewards will be estimated
        :param gamma: discount factor
        :param a_af: a set of alternative actions to be used for estimating the next-state value function
        :param ignore_episode_end: whether to ignore the last state-action pair in each trajectory. If not ignored,
            the reward for the last state-action pair will be set to the Q-value.
        :return: a tuple of tensors containing the states, actions and the corresponding the rewards
        """
        r_bc = torch.cat([self.traj_rewards(traj, gamma, a_af, ignore_last=ignore_episode_end) for traj in demos.trajectories])

        s_bf = demos.get_states_tensor(omit_episode_end=ignore_episode_end)
        a_bf = demos.get_actions_tensor(omit_episode_end=ignore_episode_end)

        if not hasattr(self, "reward_samples") or self.reward_samples is None:
            self.reward_samples = r_bc
            self.reward_eval_states = s_bf
            self.reward_eval_actions = a_bf
        else:
            self.reward_samples = torch.cat([self.reward_samples, r_bc], dim=0)
            self.reward_eval_states = torch.cat([self.reward_eval_states, s_bf], dim=0)
            self.reward_eval_actions = torch.cat([self.reward_eval_actions, a_bf], dim=0)

        return s_bf, a_bf, r_bc

    def get_reward_samples(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.reward_eval_states is not None or self.reward_eval_actions is not None or self.reward_samples is not None, \
            "Reward samples have not been calculated yet."
        return self.reward_eval_states, self.reward_eval_actions, self.reward_samples

    def train_gp(self, training_iterations=50):
        # Prepare training data
        self.train_x = torch.cat([self.reward_eval_states, self.reward_eval_actions], dim=-1)
        self.train_reward_mean = self.reward_samples.mean(dim=-1, keepdim=True)
        self.train_reward_var = self.reward_samples.var(dim=-1, keepdim=True)

        # Initialize the heteroskedastic GP model
        self.gp = botorch.models.HeteroskedasticSingleTaskGP(
            train_X=self.train_x,
            train_Y=self.train_reward_mean,
            train_Yvar=self.train_reward_var
        )

        # Find optimal model hyperparameters
        self.gp.train()
        likelihood = self.gp.likelihood

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.gp.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = ExactMarginalLogLikelihood(likelihood, self.gp)

        for i in range(training_iterations):
            optimizer.zero_grad()
            output = self.gp(self.train_x)
            loss = -mll(output, self.train_reward_mean.squeeze(-1), self.train_x)
            loss.backward()
            optimizer.step()

    def predict_gp(self, s_bf: torch.Tensor, a_bf: torch.Tensor):
        self.gp.eval()
        self.gp.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.cat([s_bf, a_bf], dim=-1)
            observed_pred = self.gp.likelihood(self.gp(test_x), test_x)
            return observed_pred

    def boltzmann_action_probs(self, s_bf: torch.Tensor, a_af: torch.Tensor, beta: float,
                               density_over_volume: Optional[float] = None, log_mean: bool = False, return_logprobs=False) -> torch.Tensor:
        """
        Returns the action probabilities according to the Boltzmann distribution.
        """
        q_vals_bac = self.q_samples(s_bf, a_af)
        return belief_weighed_action_probs(q_vals_bac, beta, density_over_volume=density_over_volume, log_mean=log_mean,
                                           return_logprobs=return_logprobs)
