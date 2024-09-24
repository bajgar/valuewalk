from envs.gridworld import ObstacleGridworldConfig, create_obstacle_gridworld, ObservationType, \
    RewardType, create_random_gridworld


def get_obstacle_gridworld_config(width, height, env_factory=create_obstacle_gridworld):
    env_config = ObstacleGridworldConfig(
        env_factory=env_factory,
        height=width,
        width=height,
        direct_access=True,
        observation_type=ObservationType.onehot,
        reward_type=RewardType.s_source
    )
    return env_config


def get_random_gridworld_config(reward_prior_factory, env_factory=create_random_gridworld, epsilon=0.1,
                               height=5, width=5, seed=None):

    env_config = ObstacleGridworldConfig(
        env_factory=env_factory,
        height=height,
        width=width,
        direct_access=True,
        observation_type=ObservationType.index,
        reward_type=RewardType.s_source,
        epsilon=epsilon,
        reward_distribution_factory=reward_prior_factory,
        seed=seed
    )
    return env_config

def get_3x3_irl_gridworld_config():
    return get_obstacle_gridworld_config(3, 3)


def get_6x6_irl_gridworld_config():
    return get_obstacle_gridworld_config(6, 6)


def get_12x12_irl_gridworld_config():
    return get_obstacle_gridworld_config(12, 12)
