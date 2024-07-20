import time
import random
from typing import Callable

import ray
from ray.rllib.policy.policy import Policy
import gymnasium as gym
import torch
import numpy as np

from configuration.configurable_factory import configurable_factory
from irl_algorithms.demonstrations import Trajectory, Demonstrations
from rl_algorithms.policies.boltzmann import TabularBoltzmannPolicy
from rl_algorithms.tabular import bellman


def collect_demos(policy: Policy, env: gym.Env, n_samples: int, rewards=False,
                  discard_incomplete=True, max_len=None, visualize=False, append_last=False,
                  ensure_tensor_states=True) -> Demonstrations:
    """
    Collects trajectories of agent acting in an environment
    :param policy:
    :param env:
    :param n_samples: number of state-action steps to collect
    :param rewards: whether to collect rewards. If True (s,a,r) triples are collected. Else, (s,a) pairs are collected.
    :param trajectories: if True, returns a list of trajectories (each a list of s-a pairs). False returns one long list
                      of s-a pairs (or s-a-r triples)
    :param discard_incomplete: if True, discards the last trajectory if it is not complete (i.e. the agent did not
                      reach a terminal state or max_len)
    :param max_len: (optional) maximum length of a single trajectory before the environment is reset. If not provided
                    trajectories are collected until.
    :return: list(list(tuple)) or list(tuple) depending on sep_trajs
    """

    demos = Demonstrations()

    n = 0
    while n < n_samples:

        s, _ = env.reset()
        terminated = False
        i = 0

        trajectory = Trajectory()

        while n < n_samples and not terminated and (max_len is None or i < max_len):

            a = policy.compute_single_action(s, None)[0]
            s1, r, terminated, truncated, _ = env.step(a)
            if rewards:
                raise NotImplementedError
            else:
                if ensure_tensor_states:
                    s = torch.tensor(s, dtype=torch.long) if isinstance(env.observation_space, gym.spaces.Discrete) else torch.tensor(s, dtype=torch.float)
                    # a = torch.tensor(a)
                trajectory.append(s, a)
            s = s1
            i += 1

            if visualize:
                env.visualize_state()
                time.sleep(0.5)

        if append_last:
            if ensure_tensor_states:
                s = torch.tensor(s, dtype=torch.float)
            trajectory.append(s, torch.zeros_like(a))

        trajectory.terminated = terminated
        trajectory.truncated = truncated

        n += i

        demos.append(trajectory)

    if discard_incomplete and not terminated and (max_len is None or len(demos[-1]) < max_len):
        demos = demos[:-1]

    return demos


@configurable_factory
def collect_demos_from_loaded_policy(demos_config: 'DemosConfig') -> Callable[[gym.Env], Demonstrations]:

    def collect_demos(env):
        ray.init()

        rl_config = demos_config.rl_config_factory()
        rl_algorithm = rl_config.build()
        rl_algorithm.load_checkpoint(demos_config.rl_checkpoint)

        observation, info = env.reset(seed=0)
        terminated = False
        truncated = False

        demos = Demonstrations()
        while len(demos) < demos_config.n_trajectories:

            trajectory = Trajectory()

            while not terminated and not truncated and len(trajectory) < rl_config.max_trajectory_len:
                action = rl_algorithm.compute_single_action(observation,
                                                    explore=False)  # this is where you would insert your policy
                next_observation, reward, terminated, truncated, info = env.step(action)

                trajectory.append(torch.tensor(observation, dtype=torch.float), action)

                observation = next_observation

            demos.append(trajectory)

            observation, info = env.reset()

        env.close()
        ray.shutdown()

        demos.save(demos_config.save_path)

        return demos

    return collect_demos


def collect_boltzmann_demos_finite(env: gym.Env, n_samples: int, beta=1., gamma=0.9, seed=None, **kwargs):

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    # value iteration -- calculates Q from rewards and transitions
    _, Q_sa = bellman.value_iteration_s_only_torch(torch.tensor(env.P_sas, dtype=torch.float),
                                                   torch.tensor(env.r_s, dtype=torch.float),
                                                   gamma=gamma,
                                                   tol=1e-6)
    # Use Q to generate a policy
    policy = TabularBoltzmannPolicy(Q_sa, observation_space=env.observation_space, action_space=env.action_space,
                                    beta=beta, onehot=False)
    # Use the policy to generate demos
    return collect_demos(policy, env, n_samples=n_samples, **kwargs)


@configurable_factory
def collect_boltzmann_demos_factory_finite(demos_config: 'DemosConfig'):

    def collect_boltzmann_demos_partial(env):
        return collect_boltzmann_demos_finite(env, n_samples=demos_config.total_samples, beta=demos_config.beta_expert,
                                              gamma=demos_config.gamma, seed=demos_config.seed,
                                              append_last=demos_config.append_last)

    return collect_boltzmann_demos_partial


@configurable_factory
def load_demos_factory(demos_config: 'DemosConfig'):

    def load_demos_partial(env):
        demos = Demonstrations.load(demos_config.demos_file_path)

        if demos_config.onehot_states_to_index:
            assert len(demos.states[0].shape) == 1
            demos = Demonstrations([Trajectory(oa_pairs=[(torch.argmax(s, dim=-1), a) for s, a in traj]) for traj in demos])

        if demos_config.index_to_onehot:
            assert len(demos.states[0].shape) == 0
            demos = Demonstrations([Trajectory(oa_pairs=[(torch.nn.functional.one_hot(s, num_classes=env.n_states).float(), a) for s, a in traj]) for traj in demos])

        if demos_config.append_last_dummy:
            last_state = torch.nn.functional.one_hot(torch.tensor(len(demos[0][0][0])-1),
                                                      num_classes=len(demos[0][0][0])).float()
            for traj in demos:
                traj.append(last_state, torch.zeros_like(traj[-1][1]))

        return demos

    return load_demos_partial


def get_boltzmann_action_probs(q_fn, state, actions, boltzmann_coeff=1.):
    """Calculate the action probabilities based on the Q function."""
    q_values = np.array([q_fn(state, action) for action in actions], dtype=np.float32)
    exp_q_values = np.exp(boltzmann_coeff * q_values)
    probabilities = exp_q_values / np.sum(exp_q_values)
    return probabilities


def sample_boltzmann_action(q_fn, state, action_space: gym.Space, boltzmann_coeff=1., num_action_samples=10):
    """Sample an action based on the Boltzmann distribution."""
    # Sample random candidate actions
    actions = [action_space.sample() for _ in range(num_action_samples)]
    probabilities = get_boltzmann_action_probs(q_fn, state, actions, boltzmann_coeff)
    # Sample an action according to the probability distribution
    action = actions[np.random.choice(len(actions), p=probabilities)]
    return np.array(action).astype(np.float32)


@configurable_factory
def collect_boltzmann_demos_factory_cts(demos_config: 'BoltzmannDemosConfig'):
    """Factory for generating demonstrations from environments that provide and (approximate) Q function
    and a sample method on their action space."""

    def boltzmann_demos_factory(env: gym.Env):
        """Generate demonstrations."""
        # Reset the environment and get initial observation

        q_fn = env.get_q_function(demos_config.gamma)

        # Initialize the list of demonstrations
        demos = Demonstrations()

        print("Started collecting demos")

        while ((demos_config.total_samples is None or len(demos) < demos_config.total_samples) and
                (demos_config.n_trajectories is None or len(demos) < demos_config.n_trajectories)):
            trajectory = Trajectory()
            observation, info = env.reset()
            done = truncated = False

            while (not done and not truncated and
                   (demos_config.total_samples is None or len(demos) < demos_config.total_samples)):
                # Sample an action from the Boltzmann distribution
                action = sample_boltzmann_action(q_fn=q_fn,
                                                 state=observation,
                                                 action_space=env.action_space,
                                                 boltzmann_coeff=demos_config.beta_expert,
                                                 num_action_samples=demos_config.num_action_samples)
                trajectory.append(torch.from_numpy(observation), torch.from_numpy(action))

                # Take a step in the environment
                observation, reward, done, truncated, info = env.step(action)
                # Add the observation and action to the list of demonstrations

            demos.append(trajectory)
            print(f"Collected trajectory with {len(trajectory)} steps")
        return demos

    return boltzmann_demos_factory
