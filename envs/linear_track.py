import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.utils import seeding

from configuration.configurable_factory import configurable_factory
from envs.env_config import EnvConfig
from irl_algorithms.demonstrations import Demonstrations, Trajectory
from irl_algorithms.demonstrations_config import BoltzmannDemosConfig
from irl_algorithms.utils.collect_trajectories import collect_boltzmann_demos_factory_cts


class LinearTrackEnv(gym.Env):
    """
    A simple linear track environment.

    The environment defines a 1D track where the agent can move along the track.
    The observation is a single number representing the position on the track.
    Actions are continuous and bounded between -1 and 1.
    The agent receives a reward of +10 if its position is 10 or more, and -1 otherwise.
    The episode terminates when the position is 10 or more.
    """

    def __init__(self, goal_reward: float = 1., time_penalty: float = 0.):
        self.min_position = -10
        self.max_position = 11
        self.goal_position = 10

        self.goal_reward = goal_reward
        self.time_penalty = time_penalty

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.min_position, high=self.max_position, shape=(1,), dtype=np.float32)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Update the position based on the action
        self.position = self.position + action[0]

        # Ensure that the position remains within bounds
        self.position = np.clip(self.position, self.min_position, self.max_position)

        # Calculate the reward
        reward = self.goal_reward if self.position >= self.goal_position else self.time_penalty

        # Check if the episode is done
        done = self.position >= self.goal_position
        truncated = False

        # Set placeholder for info
        info = {}

        # Return step information
        return np.array([self.position]).astype(np.float32), reward, done, truncated, info

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        self.position = self.np_random.normal()
        return np.array([self.position]).astype(np.float32), {}

    def get_q_function(self, gamma=0.9):
        def q_function(s, a):
            steps_to_goal = np.floor(self.goal_position + 1. - s - a)
            return self.goal_reward * gamma ** steps_to_goal - \
                   self.time_penalty * (1-gamma**steps_to_goal) / (1-gamma)

        return q_function


def get_linear_track_env_config():
    env_config = EnvConfig(
        env_factory=lambda: LinearTrackEnv(),
        obs_dim=1,
        a_dim=1,
    )
    return env_config


get_linear_track_demos_factory = collect_boltzmann_demos_factory_cts


# Test the environment
if __name__ == '__main__':
    # Create the environment
    env = LinearTrackEnv()

    # Reset the environment and get initial observation
    observation = env.reset()
    total_reward = 0
    done = False

    # Take 10 random steps in the environment
    for step in range(10):
        action = env.action_space.sample()  # Take a random action
        observation, reward, done, info = env.step(action)
        total_reward += reward
        print(f"Step {step + 1}: Action: {action}, Observation: {observation}, Reward: {reward}, Done: {done}")
        if done:
            print(f"Episode finished after {step + 1} steps with total reward {total_reward}.")
            break
    env.close()
