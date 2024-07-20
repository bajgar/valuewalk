from typing import Dict, Any

import pyro
import pyro.distributions as dist
import torch

from envs.gridworld import FiniteMDP
from irl_algorithms.demonstrations import Demonstrations
from irl_algorithms.irl_method import IRLMethod
from irl_algorithms.mcmc_irl import BayesianIRLConfig, get_pyro_mcmc_kernel
from models.reward_models.linear_reward_model import MonteCarloLinearRewardModel
from rl_algorithms.tabular.value_iteration import ValueIteration

VW_S_REWARD_KEY = 'r_s'
VW_V_KEY = 'v_s'


def valuewalk_prior_states(v_s: torch.Tensor,
                           P_sas: torch.Tensor,
                           r_prior: torch.distributions.Distribution,
                           beta: float = 10,
                           gamma: float = 0.9,
                           zero_final_reward: bool = True,
                           determinant_correction: bool = False) -> (torch.Tensor, torch.Tensor):
    """
    Calculate the log prior of a state value vector and the corresponding reward vector under a transition model.
    Args:
        v_s: vector of state values
        P_sas: transition matrix
        r_prior: prior over rewards
        beta: inverse temperature coefficient for a soft version of the optimal policy
        gamma: discount factor
        zero_final_reward: If True, the last state is supposed to be a dummy post-termination state with zero reward
        determinant_correction: If True, the log prior is corrected by the log of the absolute determinant of the
                                Bellman operator

    Returns:
        log prior of the state values and the corresponding reward vector

    """

    if zero_final_reward:
        v_s = torch.cat([v_s, torch.zeros(1, dtype=torch.float, device=P_sas.device)], dim=0)

    n_s = v_s.shape[0]

    q_sa = P_sas @ v_s
    pi_sa = torch.softmax(beta * q_sa, dim=-1)

    joint_transition_ss = torch.sum(pi_sa[:, :, None] * P_sas, dim=1)

    rew_bellman_ss = torch.eye(n_s, device=P_sas.device) - gamma * joint_transition_ss
    r_s = rew_bellman_ss @ v_s

    if zero_final_reward:
        r_s = r_s[:-1]

    v_logprior = r_prior.log_prob(r_s)

    if determinant_correction:
        abs_det = torch.abs(torch.det(rew_bellman_ss))
        v_logprior -= torch.log(abs_det)

    return v_logprior, r_s


class ValueWalkPrior(dist.Distribution):

    support = dist.constraints.real_vector

    def __init__(self, reward_prior: torch.distributions.Distribution, P_sas: torch.Tensor,
                 beta_opt: float = 10., gamma: float = 0.9, zero_final_reward: bool = True):
        """
        Prior over state values implied by reward_prior and transition model P_sas.

        Args:
            reward_prior: Prior over rewards
            P_sas: Transition matrix
            beta_opt: Inverse temperature coefficient for approximating the optimal policy
            gamma: Discount factor
            zero_final_reward:
        """

        n_states = P_sas.shape[0]
        if zero_final_reward:
            n_states -= 1
        self.r_prior = reward_prior

        self.P_sas = P_sas
        self.beta = beta_opt
        self.gamma = gamma
        self.zero_final_reward = zero_final_reward

        self.device = P_sas.device

        self.last_reward = None

    def sample(self, sample_shape=torch.Size()):
        """
        This is used to obtain the first sample for MCMC. Samples a reward from the prior and then solves value iteration
        to get the first value (similarly to PolicyWalk, except that it needs to be done only once).
        Args:
            sample_shape:

        Returns:

        """
        r_s = self.r_prior.sample(sample_shape)
        if self.zero_final_reward:
            r_s = torch.cat([r_s, torch.zeros(1, dtype=torch.float, device=self.device)], dim=0)
        q_sa = ValueIteration().run(self.P_sas, r_s, gamma=self.gamma)
        v_s = torch.max(q_sa, dim=1)[0]

        if self.zero_final_reward:
            v_s = v_s[:-1]

        self.last_reward = (r_s if not self.zero_final_reward else r_s[:-1]).detach()

        return v_s.detach()

    def log_prob(self, v_s: torch.Tensor) -> torch.Tensor:
        log_prob, r_s = valuewalk_prior_states(v_s, self.P_sas, r_prior=self.r_prior, beta=self.beta, gamma=self.gamma,
                                               zero_final_reward=self.zero_final_reward)
        self.last_reward = r_s.detach()

        return log_prob


def value_walk_model(s_t: torch.Tensor, a_t: torch.Tensor, P_sas: torch.Tensor, v_prior: ValueWalkPrior,
                     beta_expert: float = 1.,
                     gamma=0.9, zero_final_reward=True):
    """
    Pyro model of the ValueWalk algorithm for tabular MDPs.

    :param s_t: integer tensor of demonstration states
    :param a_t: integer tensor of demonstration actions
    :param P_sas: transition matrix
    :param v_prior: prior over state values
    :param beta_expert: softmax coefficient for the Boltzmann-rational expert policy
    :param gamma: discount factor
    :param zero_final_reward: should be True if we enforce a 0 reward on the terminal state
    :return:
    """

    v_s = pyro.sample(VW_V_KEY, v_prior)

    if zero_final_reward:
        P_sas = P_sas[:-1, :, :-1]

    # Calculate corresponding Q values using the transition model
    q_sa = v_prior.last_reward[:, None] + gamma * P_sas @ v_s

    # Boltzmann rational likelihood
    likelihood = pyro.distributions.Categorical(logits=beta_expert*q_sa[s_t, :])

    with pyro.plate('data', s_t.shape[0]):
        return pyro.sample('obs', likelihood, obs=a_t)


class ValueWalkTabular(IRLMethod):

    def __init__(self, env: FiniteMDP, config: BayesianIRLConfig):
        assert env.direct_access, f"{self.__class__.__name__} requires direct access to the transition dynamics"
        super().__init__(env, config)

        self.reward_prior = config.reward_prior_factory()

    def run(self, demonstrations_t: Demonstrations) -> (MonteCarloLinearRewardModel, Dict[str, Any]):

        P_sas = torch.tensor(self.env.P_sas, dtype=torch.float, requires_grad=False)
        v_prior = ValueWalkPrior(reward_prior=self.reward_prior, P_sas=P_sas, gamma=self.config.gamma)

        mcmc_kernel = get_pyro_mcmc_kernel(value_walk_model, self.config)

        mcmc = pyro.infer.MCMC(mcmc_kernel,
                               num_samples=self.config.num_samples,
                               warmup_steps=self.config.warmup_steps,
                               num_chains=self.config.num_chains,
                               disable_validation=False)

        mcmc.run(s_t=demonstrations_t.states_tensor.long(),
                 a_t=demonstrations_t.actions_tensor.squeeze(-1),
                 v_prior=v_prior,
                 P_sas=P_sas,
                 beta_expert=self.config.beta_expert,
                 gamma=self.config.gamma,
                 zero_final_reward=self.config.zero_final_reward)

        samples = mcmc.get_samples(group_by_chain=False)

        return MonteCarloLinearRewardModel(param_samples=samples[VW_V_KEY]), {}
