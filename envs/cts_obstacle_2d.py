from typing import Callable, Tuple, Dict
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import scipy

from envs.env_config import EnvConfig

GOAL_REWARD = 10.
TIME_PENALTY = -1.
OBSTACLE_REWARD = -50


def reward_fn(s, a):
    target_state = s + a
    if np.all(target_state >= 9):
        return GOAL_REWARD

    if np.all(np.logical_and(target_state >= 4, target_state <= 6)) or not np.all(np.logical_and(target_state >= -10, target_state <= 10)):
        return OBSTACLE_REWARD

    return TIME_PENALTY


class CtsObstacle2D(gym.Env):
    def __init__(self, noise_std: float = 0.1, max_steps: int = 100):
        super(CtsObstacle2D, self).__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)
        self.noise_std = noise_std
        self.state = None
        self.max_steps = max_steps
        self.current_step = 0

    def goal_reached(self) -> bool:
        return np.all(self.state >= 9)

    def in_hazardous_region(self) -> bool:
        return np.all(np.logical_and(self.state >= 4, self.state <= 6))

    def out_of_bounds(self) -> bool:
        return not np.all(np.logical_and(self.state >= -10, self.state <= 10))

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:

        action = np.clip(action, self.action_space.low, self.action_space.high)

        noise = np.random.normal(0, self.noise_std, size=2)
        self.state = self.state + action + noise
        self.current_step += 1

        reward = TIME_PENALTY
        done = False

        if self.goal_reached():
            reward = GOAL_REWARD
            done = True
        elif self.in_hazardous_region() or self.out_of_bounds():
            reward = OBSTACLE_REWARD

        truncated = not done and self.current_step >= self.max_steps

        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)

        return self.state, reward, done, truncated, {}

    def reset(self, *, seed: int = None, info: Dict = None) -> (np.ndarray, Dict):

        if seed is not None:
            np.random.seed(seed)

        self.state = np.random.normal(0, 1, size=2)
        self.current_step = 0
        return self.state, {}

    def render(self, mode: str = 'human'):
        pass  # Implement visualization if desired

    def plot_heatmap(self, func: Callable[[np.ndarray, np.ndarray], float], vmin=None, vmax=None):
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        actions = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1])).T.reshape(-1, 2)
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 10, 100)
        X, Y = np.meshgrid(x, y)

        for i, ax in enumerate(axs.flat):
            Z = np.array([[func(np.array([xx, yy]), actions[i]) for xx in x] for yy in y])
            c = ax.pcolormesh(X, Y, Z, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(f"Action: {actions[i]}")
            fig.colorbar(c, ax=ax)

        plt.tight_layout()
        plt.show()

    def get_q_function(self, gamma=0.9):
        return lambda s, a: approx_q_function(s, a, gamma=gamma)


def dist_to_goal(state: np.ndarray) -> float:

    goal = np.array([9., 9.])
    if np.all(state >= 9):
        return 0

    if state[0] > 9:
        return 9.- state[1]
    if state[1] > 9:
        return 9.- state[0]

    if state[0] > 6 or state[1] > 6 or state[1] > 3*state[0]/5 + 18/5 or state[1] < 5*state[0]/3 - 6 or (state[0]+state[1] > 10):
        return np.linalg.norm(np.array([9., 9.]) - state)

    if state[1] >= state[0] and state[1] <= 3*state[0]/5 + 18/5:
        return np.linalg.norm(np.array([4., 6.]) - state) + np.linalg.norm(np.array([9., 9.]) - np.array([4., 6.]))

    if state[1] <= state[0] and state[1] >= 5*state[0]/3 - 6:
        return np.linalg.norm(np.array([6., 4.]) - state) + np.linalg.norm(np.array([9., 9.]) - np.array([6., 4.]))

    raise ValueError("Should not reach here")


def dist_to_obstacle(state: np.ndarray) -> float:
    # Directed distance to obstacle (negative on the inside)

    if np.all(np.logical_and(state >= 4, state <= 6)):
        return - min(6-state[0], state[0]-4, 6-state[1], state[1]-4)

    # Distances to corners
    if state[0] < 4 and state[1] < 4:
        return np.linalg.norm(np.array([4., 4.]) - state)
    if state[0] > 6 and state[1] < 4:
        return np.linalg.norm(np.array([6., 4.]) - state)

    if state[0] < 4 and state[1] > 6:
        return np.linalg.norm(np.array([4., 6.]) - state)
    if state[0] > 6 and state[1] > 6:
        return np.linalg.norm(np.array([6., 6.]) - state)

    if state[0] >= 4 and state[0] <= 6 and state[1] < 4:
        return 4 - state[1]
    if state[0] >= 4 and state[0] <= 6 and state[1] > 6:
        return state[1] - 6
    if state[1] >= 4 and state[1] <= 6 and state[0] < 4:
        return 4 - state[0]
    if state[1] >= 4 and state[1] <= 6 and state[0] > 6:
        return state[0] - 6

    raise ValueError("Should not reach here")


def dist_to_edge(state: np.ndarray) -> float:
    return min(10-np.max(state), np.min(state) + 10)


def approx_q_function(state: np.ndarray, action: np.ndarray, gamma=0.8, noise_std=0.1) -> float:

    target_state = state + action

    if np.all(np.logical_and(target_state >= 9, target_state <= 10)):
        return GOAL_REWARD

    # The penalty for the risk of landing in a hazardous region dirtily approximated using a 1D Gaussian
    q = (gamma ** dist_to_goal(target_state)*GOAL_REWARD + TIME_PENALTY*(1-gamma**(dist_to_goal(target_state)+1))/(1-gamma) + OBSTACLE_REWARD * scipy.stats.norm.cdf(-dist_to_obstacle(target_state), scale=noise_std) +
         OBSTACLE_REWARD * scipy.stats.norm.cdf(-dist_to_edge(target_state), scale=noise_std))

    return q


def get_cts_obstacle_2d_env_config():
    env_config = EnvConfig(
        env_factory=lambda: CtsObstacle2D(),
        obs_dim=2,
        a_dim=2,
    )
    return env_config


if __name__ == '__main__':

    # Example usage of plot_heatmap
    def example_function(state: np.ndarray, action: np.ndarray) -> float:
        # return -np.linalg.norm(state) + np.linalg.norm(action)
        return dist_to_goal(state + action)

    env = CtsObstacle2D()
    env.plot_heatmap(approx_q_function)
