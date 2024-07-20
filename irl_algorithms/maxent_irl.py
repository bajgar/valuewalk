import torch

from irl_algorithms.irl_method import IRLMethod, IRLConfig, RewardModel
from irl_algorithms.demonstrations import Demonstrations
from envs.gridworld import FiniteMDP
from models.reward_model.linear_reward_model import LinearStateRewardModel


class MaxEntIRL(IRLMethod):

    def __init__(self, env: FiniteMDP, config: IRLConfig):
        assert env.direct_access, "MaxEntIRL requires direct access to the transition dynamics"
        super().__init__(env, config)

        self.weight_decay = config.weight_decay

    def expected_state_frequencies(self, r_f, T=10):
        """
        r_f: reward coefficient associated with each feature
        """
        P_sas = torch.from_numpy(self.env.P_sas)
        Z_s = torch.ones(self.env.n_states, dtype=torch.float)
        # Z_s[self.env.terminals] = 1.
        r_f = torch.cat([r_f, torch.zeros(1)])
        for _ in range(T):
            Z_sa = torch.sum(P_sas * Z_s[None, None, :], dim=-1) * torch.exp(r_f)[:, None]
            Z_s = torch.sum(Z_sa, dim=-1)
        pi_sa = Z_sa / Z_s[:, None]
        D_s = torch.tensor(self.env.rho0, dtype=torch.float)
        cum_D_s = D_s
        for t in range(T):
            D_s = torch.sum(torch.sum(D_s[:, None, None] * pi_sa[:, :, None] * self.env.P_sas, dim=1), dim=0)
            cum_D_s += D_s

        cum_D_s[-1] = 0

        return cum_D_s

    def expected_state_frequencies_log(self, r_f, T=10):
        """
        A more numerically stable version of expected_state_frequencies
        :param r_f:
        :param T:
        :return:
        """
        lP_sas = torch.log(torch.from_numpy(self.env.P_sas))
        lZ_s = torch.zeros(self.env.n_states, dtype=torch.float)
        for _ in range(T):
            lZ_sa = torch.logsumexp(lP_sas + lZ_s[None, None, :], dim=-1) + r_f[:, None]
            lZ_s = torch.logsumexp(lZ_sa, dim=-1)
        pi_sa = torch.exp(lZ_sa - lZ_s[:, None])
        D_s = torch.tensor(self.env.rho0, dtype=torch.float)
        cum_D_s = D_s
        for t in range(T):
            D_s = torch.sum(torch.sum(D_s[:, None, None] * pi_sa[:, :, None] * self.env.P_sas, dim=1), dim=0)
            cum_D_s += D_s

        cum_D_s[-1] = 0

        return cum_D_s

    def run(self, trajectories_t: Demonstrations, max_iters: int = 10000, lr=0.2, tol=1e-5) -> RewardModel:

        avg_feature_counts_f = trajectories_t.average_feature_sum(onehotify=True)

        r_0 = torch.randn_like(avg_feature_counts_f)
        # r_0[-1] = 0.

        r_s = r_0
        for i in range(max_iters):
            grad = avg_feature_counts_f - self.expected_state_frequencies(r_s, T=20)[:-1] - self.weight_decay * 2 * r_s
            # current_lr = lr / (1 + i/1000)
            current_lr = lr
            r_s += current_lr * grad
            # r_s[-1] = 0

            if self.config.verbose and i % 100 == 0:
                print(i)
                print(torch.norm(grad))
                print(grad)
                print(r_s)

            if torch.norm(grad) < tol:
                break

        return LinearStateRewardModel(self.env.observation_space, r_s)
