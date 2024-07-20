from functools import partial

import pyro
import pyro.distributions as dist
import torch

from envs.gridworld import FiniteMDP, ObservationType, create_inducing_points
from irl_algorithms.demonstrations import Demonstrations
from irl_algorithms.irl_method import IRLMethod
from irl_algorithms.mcmc_irl import BayesianIRLConfig
from models.components.kernels import rbf_torch
from models.reward_model.kernel_q_model import kernel_q_model
from models.reward_model.linear_reward_model import MonteCarloLinearRewardModel


VW_Q_PARAM_KEY = 'theta_q'


class QParamPriorDiscrete(dist.Distribution):

    support = dist.constraints.real_vector

    def __init__(self, n_params, r_prior, q_model, C, C_neighbour, P_c_cplus_sas, device, use_cheating_sample=True, cheating_sample_var=100,
                 beta=10, gamma=0.9, zero_terminal=True):
        """

        :param n_params:
        :param r_prior_factory:
        :param q_model:
        :param C: set of points over which the reward prior is evaluated
        :param C_neighbour: set of points that are (1) not in C but (2) are possible successors of points in C
        :param P_c_cplus_sas: s-a-s' transition probabilities for the points in C_plus=C U C_neighbour
        :param device: Torch device (cpu or cuda).
        :param use_cheating_sample: Instead of sampling via the reward prior, sample from a normal distribution with
            variance cheating_sample_var. Should be fine to use in conjunction in MCMC where it's used to initialize.
        :param cheating_sample_var:
        :param beta: Beta of the softmax used to approximate the Bellman optimality operator.
        :param gamma: Discount factor.
        :param zero_terminal: If true, assumes there's a terminal state with reward and value 0.
        """

        self.r_prior_c = r_prior

        self.q_model = q_model
        self.C = C
        self.C_neighbour = C_neighbour
        self.P_c_plus_sas = P_c_cplus_sas
        self.beta = beta
        self.gamma = gamma
        self.zero_terminal = zero_terminal

        self.device = device

        self._fake_sampling_dist = dist.Normal(torch.zeros(n_params, device=device), cheating_sample_var*torch.ones(n_params, device=device))
        self._use_cheating_sample = use_cheating_sample

    def sample(self, sample_shape=torch.Size()):
        # r_s = self.r_prior.sample(sample_shape)
        # if self.zero_terminal:
        #     r_s = torch.cat([r_s, torch.zeros(1, dtype=torch.float, device=self.device)], dim=0)
        # v_s, _ = bellman.value_iteration_s_only_torch(self.P_sas, r_s, gamma=self.gamma)
        #
        # if self.zero_terminal:
        #     v_s = v_s[:-1]
        if self._use_cheating_sample:
            q_params = self._fake_sampling_dist.sample(sample_shape)
        else:
            raise NotImplementedError
        return q_params

    def log_prob(self, theta_q: torch.Tensor) -> torch.Tensor:


        # if zero_terminal:
        #     v_s = torch.cat([v_s, torch.zeros(1, dtype=torch.float, device=P_sas.device)], dim=0)

        # q_sa = P_sas @ v_s

        # q_sa = self.q_model(self.C, theta_q)

        q_c_sa = self.q_model(self.C, theta_q)


        if self.C_neighbour:
            q_c_neighbour_sa = self.q_model(self.C_neighbour, theta_q)
            q_c_plus_sa = torch.cat([q_c_sa, q_c_neighbour_sa], dim=0)
        else:
            q_c_plus_sa = q_c_sa


        # if self.zero_terminal:
        #     q_c_sa = torch.cat([q_c_sa, torch.zeros((1, q_c_sa.shape[-1],), dtype=q_c_sa.dtype)], dim=0)
        #     q_c_plus_sa = torch.cat([q_c_plus_sa, torch.zeros((1, q_c_plus_sa.shape[-1],), dtype=q_c_plus_sa.dtype)], dim=0)

        pi_c_sa = torch.softmax(self.beta * q_c_sa, dim=-1)

        joint_transition_c_cplus_ss = torch.sum(pi_c_sa[:, :, None] * self.P_c_plus_sas, dim=1)

        r_c_sa = q_c_sa - self.gamma * joint_transition_c_cplus_ss @ q_c_plus_sa

        # rew_bellman_ss = torch.eye(n_s, device=P_sas.device) - gamma * joint_transition_ss
        # r_s = rew_bellman_ss @ v_s

        # if zero_terminal:
        #     r_s = r_s[:-1]

        # abs_det = torch.abs(torch.det(rew_bellman_ss))

        # print(abs_det)
        # if self.zero_terminal:
        #     r_c_sa = r_c_sa[:-1, :]

        q_logprior = self.r_prior_c.log_prob(torch.flatten(r_c_sa))

        return q_logprior


def value_walk_model_approx_discrete(s_df, a_d, q_model, theta_q_prior, beta_expert=4):

    theta_q = pyro.sample(VW_Q_PARAM_KEY, theta_q_prior)
    q_da = q_model(s_df, theta_q)
    likelihood = dist.Categorical(logits=beta_expert*q_da)

    pyro.sample('obs', likelihood, obs=a_d)

def get_q_prior_approx_discrete(P_sas, X_q_sf, kernel, env, r_prior, observation_type=ObservationType.norm_coords, device=None):

    sa_shape = P_sas.shape[:-1]
    n_actions = sa_shape[1]
    n_params = X_q_sf.shape[0] * n_actions


    C_is = torch.arange(P_sas.shape[0]-1, device=device)
    C_if = torch.tensor([env.get_obs(state=i, observation_mode=observation_type) for i in C_is],
                         device=device, dtype=torch.float)

    P_c_cplus_sas = torch.tensor(P_sas[:-1, :, :-1], dtype=torch.float, device=device)

    q_prior = QParamPriorDiscrete(n_params, r_prior=r_prior,
                                  q_model=lambda s_if, theta_sa: kernel_q_model(s_if, theta_sa, X_Q_sf=X_q_sf, k=kernel,
                                                                        n_a=n_actions),
                                  C=C_if,
                                  C_neighbour=None,
                                  P_c_cplus_sas=P_c_cplus_sas,
                                  device=device,
                                  use_cheating_sample=True,
                                  cheating_sample_var=100,
                                  beta=10, gamma=0.9, zero_terminal=True)

    return q_prior


class ValueWalkApproxDiscrete(IRLMethod):

    def __init__(self, env: FiniteMDP, config: BayesianIRLConfig):
        assert env.direct_access, f"{self.__class__.__name__} requires direct access to the transition dynamics"
        super().__init__(env, config)

        self.reward_prior = config.reward_prior_factory()

    def run(self, demonstrations_t: Demonstrations) -> MonteCarloLinearRewardModel:

        a_d = demonstrations_t.actions_tensor
        s_df = demonstrations_t.states_tensor
        # s_d = torch.argmax(s_d, dim=-1)
        n_f = s_df.shape[-1]

        device = torch.device('cpu')

        # Create a set of inducing points descibed by features ranging from -1 to 1 in each) dimensions.
        X_q_sf = torch.tensor(
            create_inducing_points(self.config.approximation_grid_size),
            dtype=torch.float, device=device)
        q_approximation_kernel = partial(rbf_torch, lengthscale=self.env.lengthscale)

        theta_q_prior = get_q_prior_approx_discrete(self.env.P_sas, X_q_sf=X_q_sf, kernel=q_approximation_kernel, env=self.env,
                                                    r_prior=self.reward_prior, device=None)

        nuts_kernel = pyro.infer.NUTS(value_walk_model_approx_discrete)
        mcmc = pyro.infer.MCMC(nuts_kernel,
                               num_samples=self.config.num_samples,
                               warmup_steps=self.config.warmup_steps,
                               num_chains=self.config.num_chains,
                               disable_validation=False)

        # Convert state indices to coordinates
        # s_df = torch.tensor([self.env.get_obs(state=s, observation_mode=ObservationType.norm_coords) for s in s_d], dtype=torch.float)

        mcmc.run(s_df=s_df.to(device),
                 a_d=torch.tensor(a_d).to(device),
                 # q_model=lambda s_df, theta_q: kernel_q_model(s_df, theta_q, X_Q_sf=X_q_sf, k=q_approximation_kernel),
                 q_model=lambda s_df, theta_q: torch.sum(s_df[:, :, None] * torch.reshape(theta_q,torch.Size([1, n_f, self.env.n_actions])), dim=-2),

                 theta_q_prior=theta_q_prior,
                 beta_expert=self.config.beta_expert)

        samples = mcmc.get_samples(group_by_chain=False)

        return MonteCarloLinearRewardModel(input_space=self.env.observation_space, param_samples=samples[VW_Q_PARAM_KEY])
