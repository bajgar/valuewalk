from typing import List, Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tqdm import tqdm
import gymnasium as gym

from experiments.utils.load_avril_demos import load_avril_demo_data

"""
A PyTorch reimplementation of AVRIL, closely following the original JAX implementation:
https://github.com/XanderJC/scalable-birl/blob/main/sbirl/models.py
"""


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: int | List[int], output_dim: int):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            *([nn.Linear(input_dim, hidden_dims[0]),
                nn.ELU()] +
              [nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims) - 1)] +
              [nn.Linear(hidden_dims[-1], output_dim)])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: int | List[int], output_dim: int):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            *([nn.Linear(input_dim, hidden_dims[0]),
                nn.ELU()] +
              [nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims) - 1)] +
              [nn.Linear(hidden_dims[-1], output_dim)])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AVRIL:
    def __init__(self, states_b2f: torch.Tensor, actions_b2f: torch.Tensor, state_dim: int, action_dim: int,
                 encoder_hidden_dims: int | List[int],
                 q_network_hidden_dims: int | List[int],
                 lambda_: float = 1.,
                 lr: float = 1e-4,
                 state_only: bool = True):
        self.states_b2f = torch.from_numpy(states_b2f).float()
        self.targets = torch.from_numpy(actions_b2f).long()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lambda_ = lambda_
        self.state_only = state_only

        self.encoder = Encoder(state_dim, encoder_hidden_dims, 2 if state_only else 2 * self.action_dim)
        self.q_network = QNetwork(state_dim, q_network_hidden_dims, action_dim)

        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.q_network.parameters()), lr=lr)

    def elbo(self, states_b2f: torch.Tensor, actions_b2: torch.Tensor) -> torch.Tensor:
        """
        :param states_b2f: (batch_size, 2, state_dim) batch of state-next_state pairs
        :param actions_b2: (batch_size, 2) batch of actions taken in state-next_state pairs
        """
        # Q values for all actions
        q_values_ba = self.q_network(states_b2f[:, 0, :])
        # Q values for actions actually taken
        q_values_realized_b = q_values_ba.gather(1, actions_b2[:, :1]).squeeze()

        # Q values in the next state
        q_values_next_ba = self.q_network(states_b2f[:, 1, :])
        # Q values for actions actually taken in the next state
        q_values_next_realized_b = q_values_next_ba.gather(1, actions_b2[:, 1:]).squeeze()

        # Calculate the mean and standard deviation of the reward distribution
        if self.state_only:
            r_param_b2 = self.encoder(states_b2f[:, 0, :])
            means_b = r_param_b2[:, 0]
            log_sds_b = r_param_b2[:, 1]
        else:
            r_param_bf = self.encoder(states_b2f[:, 0, :])
            means_ba = r_param_bf[:, ::2]
            log_sds_ba = r_param_bf[:, 1::2]
            means_b = means_ba.gather(1, actions_b2[:, :1]).squeeze()
            log_sds_b = log_sds_ba.gather(1, actions_b2[:, :1]).squeeze()
        sds_b = torch.exp(log_sds_b)

        # Calculate temporal difference
        td_b = q_values_realized_b - q_values_next_realized_b
        td_loss = -Normal(means_b, sds_b).log_prob(td_b).mean()

        kl = 0.5 * (-2*log_sds_b - 1.0 + sds_b ** 2 + means_b ** 2).mean()

        pred = nn.functional.log_softmax(q_values_ba, dim=1)
        neg_log_lik = -pred.gather(1, actions_b2).mean()

        return neg_log_lik + kl + self.lambda_*td_loss

    def train(self, iters: int = 1000, batch_size: int = 64) -> None:
        losses = []

        for itr in tqdm(range(iters)):
            indices = torch.randint(0, len(self.states_b2f), (batch_size,))
            inputs = self.states_b2f[indices]
            targets = self.targets[indices]

            loss = self.elbo(inputs, targets.squeeze(-1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        plt.plot(losses[10000:])
        plt.show()

    def gym_test(self, env_name: str, test_evals=100):
        results = []
        env = gym.make(env_name)
        for t in tqdm(range(test_evals), desc="Testing"):
            observation, info = env.reset()
            done = truncated = False
            rewards = []
            while not (done or truncated):
                logit = self.q_network(torch.from_numpy(observation).float().unsqueeze(0))
                action = torch.argmax(nn.functional.softmax(logit, dim=1))
                observation, reward, done, truncated, info = env.step(int(action))
                rewards.append(reward)

            results.append(sum(rewards))
        env.close()
        mean_res = sum(results) / test_evals
        print(f"Mean Reward: {mean_res}")

        return mean_res


if __name__ == "__main__":

    num_repetitions = 10

    trajectory_nums = [1, 15]
    test_rewards = []

    for n in trajectory_nums:
        test_rewards_n = []
        for i in range(num_repetitions):
            inputs, targets, a_dim, s_dim = load_avril_demo_data("CartPole-v1",
                                                                 num_trajs=n,
                                                                 randomize_demo_order=True)

            model = AVRIL(inputs, targets, s_dim, a_dim, encoder_hidden_dims=[64, 64], q_network_hidden_dims=[64, 64],
                          state_only=True, lr=1e-4)
            model.train(iters=50000)
            test_mean_reward = model.gym_test("CartPole-v1")

            test_rewards_n.append(test_mean_reward)
            print(f"{n} demos, \t test reward: {test_mean_reward}")
        test_rewards.append(sum(test_rewards_n) / num_repetitions)
        print(f"Average test reward for {n} trajs: \t {test_rewards[-1]}")

    plt.plot(trajectory_nums, test_rewards)
    plt.xlabel("Number of Demonstrations")
    plt.ylabel(f"Test Reward (averaged across {num_repetitions} runs of AVRIL and 100 test epoisodes each)")
    plt.savefig("avril_cartpole_orig_imp2.png")
    plt.show()
