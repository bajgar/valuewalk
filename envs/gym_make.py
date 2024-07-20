import gymnasium as gym
import numpy as np
from numpy import cos, pi, sin

from configuration.configurable_factory import configurable_factory
from envs.env_config import EnvConfig


@configurable_factory
def get_gym_make_factory(env_config: 'EnvConfig'):
    """
    :param env_config: an EnvConfig object including the name of the environment to make
    :return: a callable that accepts no arguments and returns a gym.Env
    """

    def env_factory():
        env = gym.make(env_config.env_name)
        env.name = env_config.env_name
        return env

    return env_factory


def get_cartpole_env_config():
    """
    :param env_name: the name of the environment to make
    :return: an EnvConfig object
    """
    return EnvConfig(
        env_factory=get_gym_make_factory,
        env_name="CartPole-v1",
        obs_dim=4,
        a_dim=2)


def get_lunar_lander_env_config():
    """
    :param env_name: the name of the environment to make
    :return: an EnvConfig object
    """
    return EnvConfig(
        env_factory=get_gym_make_factory,
        env_name="LunarLander-v2",
        obs_dim=8,
        a_dim=4)


def get_acrobot_env_config():
    """
    :param env_name: the name of the environment to make
    :return: an EnvConfig object
    """
    return EnvConfig(
        env_factory=get_gym_make_factory,
        env_name="Acrobot-v1",
        obs_dim=6,
        a_dim=3)


def get_halfcheetah_env_config():
    """
    :param env_name: the name of the environment to make
    :return: an EnvConfig object
    """
    return EnvConfig(
        env_factory=get_gym_make_factory,
        env_name="HalfCheetah-v4",
        obs_dim=17,
        a_dim=6)


def get_hopper_env_config():
    """
    :param env_name: the name of the environment to make
    :return: an EnvConfig object
    """
    return EnvConfig(
        env_factory=get_gym_make_factory,
        env_name="Hopper-v4",
        obs_dim=11,
        a_dim=3)


def get_walker2d_env_config():
    """
    :param env_name: the name of the environment to make
    :return: an EnvConfig object
    """
    return EnvConfig(
        env_factory=get_gym_make_factory,
        env_name="Walker2d-v4",
        obs_dim=17,
        a_dim=6)


def acrobot_reward(s):
    """
    :param s: the state of the Acrobot environment
    :return: the reward
    """
    return -1 if not bool(-cos(s[0]) - cos(s[1] + s[0]) > 1.0) else 0


def cartpole_reward(s):
    """
    :param s: the state of the CartPole environment
    :return: the reward
    """
    x, x_dot, theta, theta_dot = s

    x_threshold = 2.4
    theta_threshold_radians = 12 * 2 * pi / 360

    terminated = bool(
        x < x_threshold
        or x > x_threshold
        or theta < theta_threshold_radians
        or theta > theta_threshold_radians
    )

    return 1. if not terminated else 0.


def lunar_lander_shaping(state):
    shaping = (
            -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
    )
    return shaping


def lunar_lander_reward(s, a, s_prev=None):
    """
    :param s: the state of the LunarLander environment
    :return: the reward
    """
    x, y, v_x, v_y, theta, omega, l_leg, r_leg = s

    shaping = lunar_lander_shaping(s)
    if s_prev is not None:
        shaping -= lunar_lander_shaping(s_prev)

    reward = shaping

    if a == 2:
        reward -= 0.3
    elif a == 1 or a == 3:
        reward -= 0.03

    return 1. if not bool(y > 0) else 0.
