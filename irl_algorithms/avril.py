import datetime
import logging
from typing import List, Union, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tqdm import tqdm, trange
import gymnasium as gym
import numpy as np

from irl_algorithms.demonstrations import Demonstrations
from irl_algorithms.irl_method import IRLMethod
from models.reward_models.variational_reward_model import NormalVariationalRewardModel
from models.basic_models import MLP

"""
A PyTorch reimplementation of AVRIL, closely following the original JAX implementation:
https://github.com/XanderJC/scalable-birl/blob/main/sbirl/models.py
"""


if torch.cuda.is_available():
    default_device = torch.device('cuda')
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    default_device = torch.device('cpu')


class AVRIL(IRLMethod):
    def __init__(self, env: gym.Env, config: 'BayesianIRLConfig'):
        super(AVRIL, self).__init__(env, config)

        assert len(env.observation_space.shape) == 1
        self.obs_dim = env.observation_space.shape[0]

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_dim = env.action_space.n
            self.discrete_actions = True

            self.q_network = MLP(self.obs_dim, config.q_model_hidden_layer_sizes, self.action_dim)
            self.encoder = MLP(self.obs_dim, config.q_model_hidden_layer_sizes,
                               2 if self.config.state_only else 2 * self.action_dim)

        else:
            assert len(env.action_space.shape) == 1
            self.action_dim = env.action_space.shape[0]
            self.discrete_actions = False
            self.q_network = MLP(self.obs_dim+self.action_dim, config.q_model_hidden_layer_sizes, 1)
            self.encoder = MLP(input_dim=self.obs_dim if config.state_only else self.obs_dim+self.action_dim,
                               hidden_dims=config.q_model_hidden_layer_sizes,
                               output_dim=2)

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.q_network.parameters()),
                                    lr=config.svi_lr)
        self.prior_log_scale = torch.log(torch.tensor(self.config.prior_scale, dtype=torch.float))

        if self.discrete_actions:
            self.q_vals_and_r_params = self.q_vals_and_r_params_discrete
            self.select_action = self.select_discrete_action
        else:
            self.q_vals_and_r_params = self.q_vals_and_r_params_cts
            self.select_action = self.select_continuous_action
            self.action_sampler = lambda n: torch.from_numpy(np.stack([env.action_space.sample()
                                                             for _ in range(self.config.num_action_samples)])).float()

    def q_vals_and_r_params_discrete(self, states_b2f: torch.Tensor, actions_b2: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        :param states_b2f: (batch_size, 2, obs_dim) batch of state-next_state pairs
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
        if self.config.state_only:
            r_param_b2 = self.encoder(states_b2f[:, 0, :])
            means_b = r_param_b2[:, 0]
            log_sds_b = r_param_b2[:, 1]
        else:
            r_param_bf = self.encoder(states_b2f[:, 0, :])
            means_ba = r_param_bf[:, ::2]
            log_sds_ba = r_param_bf[:, 1::2]
            means_b = means_ba.gather(1, actions_b2[:, :1]).squeeze()
            log_sds_b = log_sds_ba.gather(1, actions_b2[:, :1]).squeeze()

        return q_values_ba, q_values_realized_b, q_values_next_realized_b, means_b, log_sds_b

    def q_vals_and_r_params_cts(self, states_b2f: torch.Tensor, actions_b2f: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        :param states_b2f: (batch_size, 2, obs_dim) batch of state-next_state pairs
        :param actions_b2_: (batch_size, 2) batch of actions taken in state-next_state pairs
        """

        alt_actions_af = self.action_sampler(self.config.num_action_samples)
        if len(alt_actions_af.shape) == 1:
            alt_actions_af = alt_actions_af[:, None]

        alt_actions_baf = torch.broadcast_to(alt_actions_af[None, :, :], states_b2f.shape[:-2] + alt_actions_af.shape)
        all_actions_baf = torch.cat([actions_b2f[:, 0, None, :], alt_actions_baf], dim=-2)

        s_baf = torch.broadcast_to(states_b2f[:, 0, None, :], all_actions_baf.shape[:-1] + states_b2f.shape[-1:])

        x_baf = torch.cat([s_baf, all_actions_baf], dim=-1)

        # Q values for all actions
        q_values_ba = self.q_network(x_baf).squeeze(-1)
        # Q values for actions actually taken
        q_values_realized_b = q_values_ba[:, 0]

        # Q values in the next state
        x_next_bf = torch.cat([states_b2f[:, 1, :], actions_b2f[:, 1, :]], dim=-1)
        q_values_next_realized_b = self.q_network(x_next_bf).squeeze(-1)

        # Calculate the mean and standard deviation of the reward distribution
        if self.config.state_only:
            r_param_b2 = self.encoder(states_b2f[:, 0, :])
        else:
            r_param_b2 = self.encoder(x_baf[:, 0, :])
        means_b = r_param_b2[:, 0]
        log_sds_b = r_param_b2[:, 1]

        return q_values_ba, q_values_realized_b, q_values_next_realized_b, means_b, log_sds_b

    def kl_everywhere(self, n_states=10):
        all_states = torch.eye(n_states)
        r_params = self.encoder(all_states)
        means = r_params[:, 0]
        log_sds = r_params[:, 1]
        kl = (self.prior_log_scale - log_sds +
                0.5 * (log_sds.exp() ** 2 + means ** 2)/self.config.prior_scale**2).mean()
        return kl

    def td_everywhere(self, P_sas: torch.Tensor, n_states=10):
        all_states = torch.eye(n_states)
        r_params = self.encoder(all_states)
        means = r_params[:, 0]
        log_sds = r_params[:, 1]
        sds = torch.exp(log_sds)

        q_vals_sa = self.q_network(all_states)
        v_s = torch.max(q_vals_sa, dim=1)[0]
        vals_next_sa = P_sas @ v_s
        td = vals_next_sa - self.config.gamma * v_s
        td_loss = -Normal(means, sds).log_prob(td).mean()
        return td_loss


    def elbo(self, states_b2f: torch.Tensor, actions_b2_: torch.Tensor) -> torch.Tensor:
        q_values_ba, q_values_realized_b, q_values_next_realized_b, means_b, log_sds_b = self.q_vals_and_r_params(states_b2f, actions_b2_)

        sds_b = torch.exp(log_sds_b)

        # Calculate temporal difference
        td_b = q_values_realized_b - self.config.gamma * q_values_next_realized_b
        td_loss = -Normal(means_b, sds_b).log_prob(td_b).mean()


        kl = (self.prior_log_scale - log_sds_b +
              0.5 * (sds_b ** 2 + means_b ** 2)/self.config.prior_scale**2).mean() - 0.5

        action_logits_ba = nn.functional.log_softmax(self.config.beta_expert * q_values_ba, dim=1)
        if self.discrete_actions:
            neg_log_lik = -action_logits_ba.gather(1, actions_b2_[:, :1]).mean()
            # The following is a hack to keep the last state Q vals at 0
            last_state = torch.nn.functional.one_hot(torch.tensor(states_b2f.shape[-1]-1),
                                                     num_classes=states_b2f.shape[-1]).float()
            last_state_penalty = self.config.last_state_q_penalty * (torch.abs(self.q_network(last_state))).mean()
        else:
            neg_log_lik = -action_logits_ba[:, 0].mean()
            last_state_penalty = 0

        return neg_log_lik + kl + self.config.constraint_weight * td_loss + last_state_penalty

    def train(self, states_b2f: torch.Tensor, actions_b2f: torch.Tensor,
              epochs: int = 30, batch_size: int = 64,
              valid_states_b2f: torch.Tensor = None, valid_actions_b2d: torch.Tensor = None) -> Dict[str, List]:

        results = {
            'epoch': [],
            'iter': [],
            'test_rewards': [],
            'mean_test_reward': [],
            'train_loss': [],
            'valid_loss': [],
        }

        num_data = states_b2f.shape[0]
        num_batches = num_data // batch_size

        if num_batches == 0:
            logging.warning(f"Batch size {batch_size} is larger than the number of data points {num_data}. Setting batch size to {num_data}.")
            batch_size = num_data
            num_batches = 1

        for epoch in range(epochs):
            epoch_start_time = datetime.datetime.now()

            # Shuffle the data
            permutation = torch.randperm(num_data)

            train_losses = []
            for itr in range(num_batches):
                indices = permutation[itr * batch_size: (itr + 1) * batch_size]
                inputs = states_b2f[indices]
                targets = actions_b2f[indices]

                self.optimizer.zero_grad()

                loss = self.elbo(inputs, targets)
                loss.backward()
                self.optimizer.step()

                # Record the loss
                train_losses.append(loss.item())

            if (epoch+1) % self.config.svi_reporting_frequency:
                continue

            epoch_train_time = datetime.datetime.now() - epoch_start_time

            # Record results after each epoch
            results['epoch'].append(epoch)
            results['iter'].append((epoch + 1) * num_batches)
            results['train_loss'].append(sum(train_losses) / len(train_losses))

            self.encoder.eval()
            self.q_network.eval()

            results['test_rewards'].append(self.gym_test(self.env, test_evals=self.config.test_episodes))
            results['mean_test_reward'].append(sum(results['test_rewards'][-1]) / len(results['test_rewards'][-1]))
            if valid_states_b2f is not None:
                # Switch to evaluation mode
                valid_loss = self.elbo(valid_states_b2f, valid_actions_b2d)
                print(f"Epoch: {epoch+1}, Iter: {results['iter'][-1]}, Train loss: {results['train_loss'][-1]:.4f}, Validation Loss: {valid_loss.item():.4f}, Epoch time: {epoch_train_time}")
                results['valid_loss'].append(valid_loss.item())
                # Switch back to training mode
            else:
                print(f"Epoch: {epoch+1}, Iter: {results['iter'][-1]}, Train loss: {results['train_loss'][-1]:.4f}, Test Reward: {results['mean_test_reward'][-1]:.4f}, Epoch time: {epoch_train_time}")

            self.encoder.train()
            self.q_network.train()

        return results

    def run(self, demonstrations: Demonstrations, valid_demonstrations: Demonstrations = None, eval_points=None):
        state_next_state, action_next_action = demonstrations.get_two_step_pairs()
        state_next_state = state_next_state.to(default_device).float()
        action_next_action = action_next_action.to(default_device).float()

        if self.discrete_actions:
            action_next_action = action_next_action.squeeze(-1).long()

        if valid_demonstrations is not None:
            valid_state_next_state, valid_action_next_action = valid_demonstrations.get_two_step_pairs()
            if self.discrete_actions:
                valid_action_next_action = valid_action_next_action.squeeze(-1).long()
        else:
            valid_state_next_state = valid_action_next_action = None

        self.eval_points = eval_points

        results = self.train(state_next_state, action_next_action,
                   epochs=self.config.epochs, batch_size=self.config.batch_size,
                   valid_states_b2f=valid_state_next_state, valid_actions_b2d=valid_action_next_action)

        info = results
        info['test_mean_reward'] = self.gym_test(self.env)

        return NormalVariationalRewardModel(self.encoder, self.q_network), info

    def select_discrete_action(self, observation: np.ndarray) -> int:
        logit = self.q_network(torch.from_numpy(observation).float().unsqueeze(0))
        action = torch.argmax(logit)
        return int(action)

    def select_continuous_action(self, observation: np.ndarray) -> np.ndarray:
        alt_actions_af = self.action_sampler(self.config.num_action_samples)
        if len(alt_actions_af.shape) == 1:
            alt_actions_af = alt_actions_af[:, None]

        s_af = torch.broadcast_to(torch.from_numpy(observation[None, :]).float(),
                                  alt_actions_af.shape[:-1] + observation.shape[-1:])

        x_af = torch.cat([s_af, alt_actions_af], dim=-1)
        logit_a = self.q_network(x_af).squeeze(-1)
        action_ix = torch.argmax(logit_a)
        return alt_actions_af[action_ix].cpu().numpy()

    def gym_test(self, test_env: gym.Env, test_evals=300):
        start_time = datetime.datetime.now()
        results = []
        env = test_env
        for t in range(test_evals):
            observation, info = env.reset()
            done = truncated = False
            rewards = []
            while not (done or truncated):
                action = self.select_action(observation)

                observation, reward, done, truncated, info = env.step(action)
                rewards.append(reward)

            results.append(sum(rewards))
        env.close()
        mean_res = sum(results) / test_evals
        eval_time = datetime.datetime.now() - start_time
        print(f"Mean Reward: {mean_res}, Eval Time: {eval_time}")

        return results
