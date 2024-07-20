from typing import Optional, Dict, Any

import numpy as np
import scipy
import torch
import pyro

from irl_algorithms.irl_method import IRLMethod, RewardModel, IRLConfig
from irl_algorithms.demonstrations import Demonstrations
from envs.gridworld import FiniteMDP
from irl_algorithms.mcmc_irl import BayesianIRLConfig, get_pyro_mcmc_kernel
from models.reward_models.linear_reward_model import MonteCarloLinearRewardModel
from rl_algorithms.tabular.value_iteration import ValueIteration, ValueIterationNP

PW_REWARD_KEY = 'r_s'


def policy_walk_model(s_t: torch.Tensor, a_t: torch.Tensor,
                      reward_prior: pyro.distributions.Distribution,
                      P_sas: torch.Tensor,
                      value_iteration_instance: ValueIteration = None,
                      beta_expert: float = 4.,
                      gamma: float = 0.9,
                      zero_final_reward: bool = True):
    """
    Pyro model of the tabular version of the Bayesian inverse reinforcement learning problem with Boltzmann-rational
    demonstrations.

    :param s_t: integer tensor of demonstration states
    :param a_t: integer tensor of demonstration actions
    :param reward_prior: a prior over rewards
    :param P_sas: transition matrix
    :param value_iteration_instance: value iteration instance that can cache the previous value
    :param beta_expert: rationality coefficient for the Boltzmann-rational expert policy
    :param gamma: discount factor
    :param zero_final_reward: should be True if we enforce a 0 reward on the final state (supposed to represent
                              a dummy post-termination state)
    """

    # Reward vector is assumed to have been sampled from the prior
    r_s = pyro.sample(PW_REWARD_KEY, reward_prior)

    if zero_final_reward:
        # Final state has fixed zero reward
        r_s = torch.cat([r_s, torch.zeros(1)])

    # Calculate corresponding Q values using the transition model
    q_sa = value_iteration_instance.run(P_sas=P_sas, r=r_s, gamma=gamma)

    # Boltzmann rational likelihood
    likelihood = pyro.distributions.Categorical(logits=beta_expert*q_sa[s_t, :])

    with pyro.plate('data', s_t.shape[0]):
        return pyro.sample('obs', likelihood, obs=a_t)


class PolicyTracker:
    """
    Value iteration algorithm for solving MDPs, which can cache the previous value vector for a speedup if run
    repeatedly on an evolving MDP. This work for 1-D or 2-D rewards depending on the source state (not target state).
    """

    def __init__(self):
        self._last_q = None
        self._inv_bellman = None
        self._current_policy = None

    def run(self, P_sas: torch.Tensor, r: torch.Tensor,
            q_prev: Optional[torch.Tensor] = None,
            gamma: float =0.9, max_iters: int = 1000, tol: float = 1e-4) -> torch.Tensor:
        """
        :param P_sas: [n_states] x [n_actions] x [n_states] tensor where p[s0, a, s1] is the probability of
                    transitioning from s0 to s1 if action a is taken
        :param r: reward vector
        :param q_prev: previous Q values to start from
        :param gamma: discount rate
        :param max_iters: maximum number of iterations
        :param tol: tolerance for convergence

        :return: Q values
        """

        if self._inv_bellman is not None:
            V_s = self._inv_bellman @ r
            Q_sa = r[:,None] + gamma * P_sas @ V_s
            new_policy = torch.argmax(Q_sa, dim=1)
            self._last_q = Q_sa
            if torch.all(new_policy == self._current_policy):
                return Q_sa

        n_states, n_actions, _ = P_sas.shape
        if q_prev is not None:
            Q_sa = q_prev
            self._last_q = q_prev
        elif self._last_q is not None:
            Q_sa = self._last_q
        else:
            Q_sa = torch.zeros([n_states, n_actions])
            self._last_q = Q_sa.detach()

        if len(r.shape) == 1:
            r = r.unsqueeze(1)

        for it in range(max_iters):
            # Bellman update
            V_s = Q_sa.max(dim=1)[0]
            Q_sa = r + gamma * P_sas @ V_s

            # Test for convergence
            q_diff = torch.max(torch.abs(Q_sa - self._last_q))
            if q_diff < tol:
                break
            self._last_q = Q_sa.detach()

        self._current_policy = torch.argmax(Q_sa, dim=1)
        self._inv_bellman = torch.inverse(torch.eye(n_states) - gamma * P_sas[torch.arange(n_states), self._current_policy, :]).detach()

        r = r.squeeze()
        V_s = self._inv_bellman @ r
        Q_sa = r[:, None] + gamma * P_sas @ V_s

        return Q_sa


class PolicyWalk(IRLMethod):

    def __init__(self, env: FiniteMDP, config: BayesianIRLConfig):
        assert env.direct_access, f"{self.__class__.__name__} requires direct access to the transition dynamics"
        super().__init__(env, config)

        self.reward_prior = config.reward_prior_factory()

    def run(self, demonstrations_t: Demonstrations) -> (MonteCarloLinearRewardModel, Dict[str, Any]):

        value_iteration = PolicyTracker()

        mcmc_kernel = get_pyro_mcmc_kernel(policy_walk_model, self.config)

        mcmc = pyro.infer.MCMC(mcmc_kernel,
                               num_samples=self.config.num_samples,
                               warmup_steps=self.config.warmup_steps,
                               num_chains=self.config.num_chains,
                               disable_validation=False)

        mcmc.run(s_t=demonstrations_t.states_tensor.long(),
                 a_t=demonstrations_t.actions_tensor.squeeze(-1),
                 reward_prior=self.reward_prior,
                 P_sas=torch.tensor(self.env.P_sas, dtype=torch.float),
                 beta_expert=self.config.beta_expert,
                 value_iteration_instance=value_iteration)

        samples = mcmc.get_samples(group_by_chain=False)

        return MonteCarloLinearRewardModel(param_samples=samples[PW_REWARD_KEY]), {}



class PolicyWalkRamachandran(IRLMethod):
    def __init__(self, env: FiniteMDP, config: BayesianIRLConfig):
        assert env.direct_access, f"{self.__class__.__name__} requires direct access to the transition dynamics"
        super().__init__(env, config)

        self.reward_prior = config.reward_prior_factory()

        self.step_size = config.hmc_step_size
        self.num_samples = config.num_samples
        self.beta_expert = config.beta_expert
        self.gamma = config.gamma
        self.zero_final_reward = config.zero_final_reward

        # Initialize the current log-posterior and Q-values
        self.current_log_posterior = -np.inf
        self.current_q_values = None
        self.current_policy = None

        if self.zero_final_reward:
            self.P_sas = env.P_sas[:-1, :, :-1]
        else:
            self.P_sas = env.P_sas

        self.inv_bellman = None

        self.subsample = 100  # Reward samples for grid walk are highly correlated. Keep only one in 100.
        self.tol = 1e-3
        self.max_iters = 1000

    def run(self, demonstrations_t: Demonstrations) -> (MonteCarloLinearRewardModel, Dict[str, Any]):
        # Extract necessary information from demonstrations
        s_t = demonstrations_t.states_tensor.long().numpy()
        a_t = demonstrations_t.actions_tensor.squeeze(-1).numpy()

        # Initialize reward samples and optimal policy
        reward_samples = []
        optimal_policy = None

        # Initialize the current reward vector
        current_reward = self.reward_prior.sample()
        self.current_q_values = self._value_iteration(current_reward)
        self.current_policy = self._compute_optimal_policy(current_reward)
        self.inv_bellman = np.linalg.inv(np.eye(self.env.n_states-1) - self.gamma * self.P_sas[np.arange(len(self.current_policy )), self.current_policy , :])


        # Perform PolicyWalk for the specified number of samples
        for step in range(self.num_samples):

            # Sample a neighboring reward vector from the grid
            neighbor_reward = self._sample_neighbor_reward(current_reward)

            # Accept or reject the neighboring reward vector
            acceptance_prob, neighbor_log_posterior, neighbor_q_values = self._compute_acceptance_prob(
                neighbor_reward, s_t, a_t
            )

            if np.random.rand() < acceptance_prob:
                current_reward = neighbor_reward
                self.current_log_posterior = neighbor_log_posterior
                self.current_q_values = neighbor_q_values

                if not self._is_optimal_policy(neighbor_q_values, self.current_policy):
                    self.current_policy = np.argmax(neighbor_q_values, axis=1)
                    self.inv_bellman = np.linalg.inv(np.eye(self.env.n_states-1) - self.gamma * self.P_sas[np.arange(len(self.current_policy )), self.current_policy , :])

            # Store the current reward vector as a sample
            if step % self.subsample == 0:
                reward_samples.append(current_reward)

        return MonteCarloLinearRewardModel(param_samples=np.array(reward_samples)), {}

    def _compute_optimal_policy(self, reward):

        optimal_policy = np.argmax(self.current_q_values, axis=1)
        return optimal_policy

    def _sample_neighbor_reward(self, reward):
        # Sample a neighboring reward vector from the grid
        neighbor_reward = reward.copy()
        idx = np.random.randint(len(reward))
        neighbor_reward[idx] += np.random.choice([-self.step_size, self.step_size])
        return neighbor_reward

    def _compute_q_values(self, reward):
        # Compute the Q-values for the given reward vector and policy
        v_values = self.inv_bellman @ reward
        q_values = reward.reshape(-1, 1) + self.gamma * self.P_sas @ v_values
        return q_values

    def _is_optimal_policy(self, q_values, policy):
        # Check if the given policy is optimal with respect to the Q-values
        return np.allclose(np.argmax(q_values, axis=1), policy)

    def _value_iteration(self, reward):
        # Perform value iteration to find the Q-values for the given reward vector
        if self.current_q_values is None:
            q_values = np.zeros((self.env.n_states-1 if self.zero_final_reward else self.env.n_states, self.env.n_actions))
        else:
            q_values = self.current_q_values

        last_q_values = q_values
        for _ in range(self.max_iters):
            v_values = np.max(q_values, axis=1)
            q_values = reward.reshape(-1, 1) + self.gamma * self.P_sas @ v_values

            q_diff = np.max(np.abs(q_values - last_q_values))
            last_q_values = q_values

            if q_diff < self.tol:
                break

        return q_values

    def _compute_acceptance_prob(self, neighbor_reward, s_t, a_t):
        # Compute the Q-values for the neighboring reward vector
        neighbor_q_values = self._compute_q_values(neighbor_reward)

        if not self._is_optimal_policy(neighbor_q_values, self.current_policy):
            neighbor_q_values = self._value_iteration(neighbor_reward)

        # Compute the log-likelihood for the neighboring reward vector
        neighbor_log_softmax = scipy.special.log_softmax(self.beta_expert * neighbor_q_values, axis=1)
        neighbor_log_likelihood = neighbor_log_softmax[s_t, a_t].sum()

        # Compute the log-prior probability for the neighboring reward vector
        neighbor_log_prior = self.reward_prior.log_prob(neighbor_reward).sum()

        # Compute the log-posterior probability for the neighboring reward vector
        neighbor_log_posterior = neighbor_log_prior + neighbor_log_likelihood

        # Compute the acceptance probability in log-space
        log_acceptance_prob = min(0, neighbor_log_posterior - self.current_log_posterior)
        acceptance_prob = np.exp(log_acceptance_prob)

        return acceptance_prob, neighbor_log_posterior, neighbor_q_values
