from typing import List, Union, Tuple, Dict
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tqdm import tqdm, trange
import gymnasium as gym
import numpy as np

from irl_algorithms.demonstrations import Demonstrations
from irl_algorithms.irl_method import IRLMethod
from irl_algorithms.utils.gaussians import kl_between_gaussians
from models.reward_models.variational_reward_model import NormalVariationalRewardModel
from models.basic_models import MLP


if torch.cuda.is_available():
    default_device = torch.device('cuda')
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    default_device = torch.device('cpu')


class BC(IRLMethod):
    def __init__(self, env: gym.Env, config: 'BayesianIRLConfig', additional_params: List[torch.Tensor] = None):
        super(BC, self).__init__(env, config)

        self.obs_dim = env.observation_space.shape[0]

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_dim = env.action_space.n
            self.discrete_actions = True
            self.policy_net = MLP(input_dim=self.obs_dim,
                                  hidden_dims=config.q_model_hidden_layer_sizes,
                                  output_dim=self.action_dim)

        else:
            assert len(env.action_space.shape) == 1
            self.action_dim = env.action_space.shape[0]
            self.discrete_actions = False
            self.policy_net = MLP(input_dim=self.obs_dim,
                                  hidden_dims=config.q_model_hidden_layer_sizes,
                                  output_dim=self.action_dim)

        self.parameters = list(self.policy_net.parameters())

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.optimizer = optim.Adam(self.parameters,
                                    lr=config.svi_lr)

        if self.discrete_actions:
            self.select_action = self.select_discrete_action
            self.supervised_loss = self.supervised_loss_discrete
        else:
            self.select_action = self.select_continuous_action
            self.supervised_loss = self.supervised_loss_continuous

    def supervised_loss_discrete(self, states_bf: torch.Tensor, actions_bf: torch.Tensor) -> torch.Tensor:
        logits = self.policy_net(states_bf)
        return nn.functional.cross_entropy(logits, actions_bf.squeeze(-1))

    def supervised_loss_continuous(self, states_bf: torch.Tensor, actions_bf: torch.Tensor) -> torch.Tensor:
        actions_pred = self.policy_net(states_bf)
        return nn.functional.mse_loss(actions_pred, actions_bf)

    def train(self, states_bf: torch.Tensor, actions_bf: torch.Tensor,
              epochs: int = 30, batch_size: int = 64,
              valid_states_bf: torch.Tensor = None, valid_actions_bd: torch.Tensor = None) -> Dict[str, List]:

        results = {
            'epoch': [],
            'iter': [],
            'test_rewards': [],
            'mean_test_reward': [],
            'train_loss': [],
            'valid_loss': [],
        }

        num_data = states_bf.shape[0]

        if num_data < batch_size:
            batch_size = num_data

        num_batches = num_data // batch_size

        for epoch in range(epochs):
            epoch_start_time = datetime.datetime.now()

            # Shuffle the data
            permutation = torch.randperm(num_data)

            train_losses = []

            for itr in range(num_batches):
                # Select the batch
                indices = permutation[itr * batch_size: (itr + 1) * batch_size]
                inputs = states_bf[indices]
                targets = actions_bf[indices]

                # Compute loss and backpropagate
                self.optimizer.zero_grad()
                loss = self.supervised_loss(inputs, targets)
                loss.backward()
                self.optimizer.step()

                # Record the loss
                train_losses.append(loss.item())

            if (epoch + 1) % self.config.svi_reporting_frequency:
                continue

            epoch_train_time = datetime.datetime.now() - epoch_start_time

            # Record results after each epoch
            results['epoch'].append(epoch)
            results['iter'].append((epoch + 1) * num_batches)
            results['train_loss'].append(sum(train_losses) / len(train_losses))

            self.policy_net.eval()

            results['test_rewards'].append(self.gym_test(self.env, test_evals=self.config.test_episodes))
            results['mean_test_reward'].append(sum(results['test_rewards'][-1]) / len(results['test_rewards'][-1]))

            print(
                f"Epoch: {epoch + 1}, Iter: {results['iter'][-1]}, Train loss: {results['train_loss'][-1]:.4f}, Test Reward: {results['mean_test_reward'][-1]:.4f}, Epoch time: {epoch_train_time}")
            self.policy_net.train()

        self.policy_net.eval()

        return results

    def run(self, demonstrations: Demonstrations, valid_demonstrations: Demonstrations = None):
        state_next_state, action_next_action = demonstrations.get_two_step_pairs()
        state_next_state = state_next_state.to(default_device)
        action_next_action = action_next_action.to(default_device)

        if self.discrete_actions:
            action_next_action = action_next_action.squeeze(-1).long()

        if valid_demonstrations is not None:
            valid_state_next_state, valid_action_next_action = valid_demonstrations.get_two_step_pairs()
            if self.discrete_actions:
                valid_action_next_action = valid_action_next_action.squeeze(-1).long()
        else:
            valid_state_next_state = valid_action_next_action = None

        # # States are now integers. Change to onehot encoding.
        # state_next_state = torch.nn.functional.one_hot(state_next_state.long(), num_classes=self.obs_dim).float()

        results = self.train(state_next_state, action_next_action,
                             epochs=self.config.epochs, batch_size=self.config.batch_size,
                             valid_states_bf=valid_state_next_state, valid_actions_bd=valid_action_next_action)

        info = results

        return NormalVariationalRewardModel(self.policy_net), info


    def select_continuous_action(self, observation: np.ndarray) -> np.ndarray:
        observation = torch.tensor(observation, dtype=torch.float).to(default_device)
        action = self.policy_net(observation).detach().cpu().numpy()
        return action

    def gym_test(self, test_env: gym.Env, test_evals=300, max_steps=1001):
        start_time = datetime.datetime.now()
        results = []
        env = test_env
        for t in range(test_evals):
            observation, info = env.reset()
            done = truncated = False
            rewards = []
            while not (done or truncated or len(rewards) > max_steps):
                action = self.select_action(observation)
                observation, reward, done, truncated, info = env.step(action)
                rewards.append(reward)

            results.append(sum(rewards))
        env.close()
        mean_res = sum(results) / test_evals
        eval_time = datetime.datetime.now() - start_time
        print(f"Mean Reward: {mean_res}, Eval Time: {eval_time}")

        return results
