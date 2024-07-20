import re
import enum
from typing import Callable

import random
import numpy as np
import gymnasium as gym

from configuration.configurable_factory import configurable_factory
from experiments.experiment_config import ExperimentConfig
from visualisation import img_utils
from rl_algorithms.tabular.bellman import value_iteration
from envs.env_config import EnvConfig
from models.components.kernels import rbf_numpy as rbf


class RewardType(enum.Enum):
    """
    Gridworld reward types. Indicates that for an s-a-s' transition, the reward is allowed to depend only on
    s, sa, s', or sas' respectively.
    """
    s_source = 's_source'
    sa = 'sa'
    s_target = 's_target'
    sas = 'sas'


class ObservationType(enum.Enum):
    """
    Gridworld observation types
    """
    onehot = 'onehot'   # One-hot encoding of discrete states
    index = 'index'     # Integer index of the observation
    coords = 'coords'   # Pair of integer coordinates
    norm_coords = 'norm_coords'  # Pair of float coordinates, each normalized to [-1, 1]
    kernel = 'kernel'   # Kernel-approximated feature vectors with respect to a set reference inducing points.


class ObstacleGridworldConfig(EnvConfig):

    width: int
    height: int

    num_actions: int = 5  # Currently 5 is the only option for a Gridworld

    time_penalty: float = 1.  # Penalty for each time step
    goal_reward: float = 10.
    obstacle_reward: float = -30.
    epsilon: float = 0.1    # Probability of random movement at each step

    reward_type: RewardType = RewardType.s_source
    observation_type: ObservationType = ObservationType.onehot
    kernel_grid: tuple = (3, 3)     # Only used for kernel observations.

    @property
    def num_states(self):
        return self.width * self.height + 1


class FiniteMDP(gym.Env):

    def __init__(self, p=None, r=None, s0=None, terminals=(), gamma=0.9, direct_access=False, max_steps: int = False,
                 **kwargs):
        """
        :param p: |S|x|A|x|S| transition probability tensor
        :param r: reward tensor (either |S|x|A|x|S| or |S|x|A| or |S|)
        :param s0: initial state distribution (probability vector of length |S|)
        :param terminals: list of terminal states
        :param gamma: discount factor
        :param direct_access: whether to allow access to the full MDP description. Can be set to False when the
                              environment is used within a model-free method to ensure certain attributes can't be
                              accesesd.
        :param max_steps: maximum number of steps per episode
        """
        self._p = p
        self._r = r
        self._s0 = s0
        self._gamma = gamma
        self._terminals = terminals
        self.direct_access = direct_access

        self._state = None
        self._n_states, self._n_actions, _ = np.shape(self._p) if self._p is not None else (None, None, None)
        self._states = np.arange(self._n_states)    if self._n_states is not None else None

        self.observation_space = gym.spaces.Discrete(self._n_states)
        self.action_space = gym.spaces.Discrete(self._n_actions)

        self._opt_values = None
        self._q = None

        self._max_steps = max_steps
        self._episode_steps = None
        self._total_return = 0
        self._episode_return = 0

        self._done = False
        self._truncated = False

        super().__init__()

    def get_mdp_description(self) -> (np.ndarray, np.ndarray, np.ndarray, float):
        if not self.direct_access:
            raise PermissionError("Direct access to the MDP not allowed. If required, set direct_access to True.")

        return self._p, self._r, self._s0, self._gamma

    def reset(self, *, seed=None, options=None):
        self._state = np.random.choice(range(self._n_states), p=self._s0)
        self._episode_steps = 0
        self._episode_return = 0
        self._done = False
        self._truncated = False

        return self.get_obs(), {}

    def step(self, action):
        s_next = np.random.choice(self._states, p=self._p[self._state, action, :])
        if self._r.ndim == 3:
            reward = self._r[self._state, action, s_next]
        elif self._r.ndim == 2:
            reward = self._r[self._state, s_next]
        elif self._r.ndim == 1:
            reward = self._r[s_next]
        else:
            raise NotImplementedError("Reward tensor must have 1,2, or 3 dimensions.")

        self._total_return += reward
        self._episode_return += reward
        self._episode_steps += 1
        if self._max_steps and self._episode_steps >= self._max_steps:
            self._truncated = True

        self._state = s_next
        self._done = s_next in self._terminals

        return self.get_obs(), reward, self._done, self._truncated, {}

    def calculate_value_function(self):
        self._opt_values, self._q = value_iteration(self._p, self._r, self._gamma)

    def get_state(self):
        return self._state

    def get_obs(self, state=None):
        if state is None:
            state = self._state
        return state

    @property
    def done(self):
        return self._done

    @property
    def n_states(self):
        return self._n_states

    @property
    def n_actions(self):
        return self._n_actions

    @property
    def P_sas(self):
        if not self.direct_access:
            raise PermissionError("Direct access to the MDP not allowed. If required, set direct_access to True.")
        return self._p

    @property
    def r_sa(self):
        if not self.direct_access:
            raise PermissionError("Direct access to the MDP not allowed. If required, set direct_access to True.")
        return self._r

    @property
    def rho0(self):
        if not self.direct_access:
            raise PermissionError("Direct access to the MDP not allowed. If required, set direct_access to True.")
        return self._s0

    @property
    def gamma(self):
        return self._gamma

    @property
    def terminals(self):
        if not self.direct_access:
            raise PermissionError("Direct access to the MDP not allowed. If required, set direct_access to True.")
        return self._terminals


def create_inducing_points(grid_size: tuple, xlims=(-1, 1), ylims=(-1, 1)):
    """
    For kernel-based observations, create a grid of inducing points in the 2D space.
    """
    x = np.linspace(*xlims, grid_size[0])
    y = np.linspace(*ylims, grid_size[1])
    xx, yy = np.meshgrid(x, y)
    return np.array((yy.ravel(), xx.ravel())).T


class GridWorld(FiniteMDP):

    n_actions = 5
    actions = range(n_actions)
    action_vectors = ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1))

    def __init__(self, grid, epsilon: float = 0., s0_grid=None, observations=ObservationType.index,
                 reward_type=RewardType.s_source, kernel_grid_size=(3, 3), delayed_terminal=True, **kwargs):
        """

        :param grid: gridworld representation (see the description of build_from_grid)
        :param epsilon: probability of random movement
        :param s0_grid:
        :param observations:
        :param reward_type:
        :param kernel_grid_size: Only used for kernel observations. Can ignore otherwise.
        :param delayed_terminal: When True, the terminal flag is returned one step later.
        :param kwargs:
        """

        self._n_actions = 5
        self._n_cols = len(grid[0])
        self._n_rows = len(grid)
        self._n_states = self._n_cols * self._n_rows + 1
        self.reward_type = reward_type
        self.delayed_terminal = delayed_terminal
        p, self.r_s, s0, terminals = self.build_from_grid(grid, s0_grid, epsilon)

        if reward_type == RewardType.s_source:
            r = np.broadcast_to(self.r_s[:, None, None], p.shape)
        elif reward_type == RewardType.s_target:
            r = np.broadcast_to(self.r_s[None, None, :], p.shape)
        else:
            raise NotImplementedError("Not yet implemented")

        self._observation_mode = observations
        if self._observation_mode == ObservationType.kernel:
            self.inducing_points = create_inducing_points(kernel_grid_size)
            # Calculate the lengthscale as an average for the two dimensions
            mean_grid_size = np.mean(kernel_grid_size)
            # Normalized coordinates are between -1 and 1, i.e. a total length of 2
            self.lengthscale = 1 / (mean_grid_size-1)


        super().__init__(p, r, s0, terminals, **kwargs)

        self.observation_space = self.determine_obs_space()

        self._last_pos = (None, None)

    def determine_obs_space(self):
        if self._observation_mode == ObservationType.index:
            return gym.spaces.Discrete(self._n_states)
        elif self._observation_mode == ObservationType.onehot:
            return gym.spaces.Box(low=0, high=1, shape=(self._n_states,))
        elif self._observation_mode == ObservationType.kernel:
            return gym.spaces.Box(low=0, high=1, shape=(len(self.inducing_points),))

    def build_from_grid(self, grid, s0_grid=None, epsilon=0):
        """
        Converts a gridworld representation (a list of lists described below) into a standard form finite MDP, FiniteMDP,
        The Gridworld is represented as a list of lists (rows) where each cell is one of the following
        - a number - representing the reward associated with transitioning into that state
        - a string in the form "\d+[TI]" where the number denotes the reward, T indicates a terminal state and
            I indicates an initial state. This will result in a uniform distribution among initial states. Alternatively
            the initial state distribution can be provided in s0_grid
        :param grid: list of lists described above
        :param s0_grid: 2d array of initial state probabilities
        :param epsilon: random noise (prob that a random movement is made instead of the intended action)
        :return: p, r, s0, terminals where
            p is a |S|x|A|x|S| tensor of transition probabilities
            r is a reward vector
        """
        n_states = self._n_states
        p = np.zeros((n_states, self._n_actions, n_states))
        r = np.zeros((n_states,))
        s0 = np.zeros((n_states,))
        terminals = [n_states-1]

        rand_trans_prob = epsilon / 5
        intended_trans_prob = 1. - epsilon

        # Cycle through the grid and populate the transition matrix, initial state distribution, and reward vector
        for i, row in enumerate(grid):
            assert len(row) == self._n_cols

            for j, cell in enumerate(row):

                idx = self.pos2idx((i, j))
                terminal = False
                if isinstance(cell, float) or isinstance(cell, int):
                    r[idx] = cell
                elif isinstance(cell, str):
                    if re.match('^[\d.]+$', cell):
                        r[idx] = float(cell)
                    else:
                        m = re.match('^([-\d.]+)([TI])$', cell)
                        if m:
                            r[i * self._n_cols + j] = float(m.group(1))
                            if m.group(2) == "T":
                                terminal = True
                                if not self.delayed_terminal:
                                    terminals.append(i * self._n_cols + j)
                            if s0_grid is None and m.group(2) == "I":
                                s0[idx] = 1.

                for action in self.actions:
                    if terminal:
                        p[idx, action, -1] = 1
                        continue
                    target_idx = self.action_target_idx(idx, action)
                    p[idx, action, target_idx] = intended_trans_prob
                    for random_action in self.actions:
                        p[idx, action, self.action_target_idx(idx, random_action)] += rand_trans_prob

                if s0_grid is not None:
                    s0[idx] = s0_grid[i, j]
        p[-1, :, -1] = 1

        # Normalize the initial state distribution
        s0 = np.array(s0) / np.sum(s0)

        return p, r, s0, terminals

    def get_norm_coords(self, state: int):
        """
        Get normalized coordinates corresponding to state
        Args:
            state: integer index of the state

        Returns:

        """
        obs = np.array(self.idx2pos(state))
        obs = 2 * obs / np.array([self._n_rows - 1, self._n_cols - 1]) - 1
        return obs

    def get_obs(self, state: int = None, observation_mode: ObservationType = None):
        """
        Get the observation corresponding to a state (by default the current state, but optionally for a specified state)
        Args:
            state (optional): state index
            observation_mode: ObservationType to return (otherwise the env's default is returned)

        Returns:

        """
        observation_mode = observation_mode or self._observation_mode

        if state is None:
            state = self._state

        if observation_mode == ObservationType.index:
            return super(GridWorld, self).get_obs(state)
        elif observation_mode == ObservationType.coords:
            obs = np.array(self.idx2pos(state))
            return obs
        elif observation_mode == ObservationType.norm_coords:
            return self.get_norm_coords(state)
        elif observation_mode == ObservationType.kernel:
            unnormalized_weights = np.squeeze(rbf(self.get_norm_coords(state)[None, :], self.inducing_points,
                                                  self.lengthscale))
            return unnormalized_weights / np.sum(unnormalized_weights)
        elif observation_mode == ObservationType.onehot:
            obs = np.zeros((self.n_states,))
            obs[state] = 1
            return obs
        else:
            raise NotImplementedError(f"Observation mode '{observation_mode}' not implemented")

    def pos2idx(self, *args):
        """
        Converts a 2d position to a 1d index
        input: 2d position represented as a pair of integers
        returns: index of the corresponding state
        """
        if len(args) == 1:
            pos = args[0]
        else:
            pos = args

        return pos[0] * self._n_cols + pos[1]

    def idx2pos(self, idx: int):
        """
        Converts a 1D state index to a 2D position
        input:
          1d idx
        returns:
          2d row-major position
        """

        if idx == self.n_states - 1:
            return self._last_pos

        pos = (idx // self._n_cols, idx % self._n_cols)
        self._last_pos = pos

        return pos

    def action_target_idx(self, state, action):
        """
        Returns
        :param state: (int) current state index
        :param action: (int) action number
        :return: (int) intended target state
        """
        pos = self.idx2pos(state)
        candidate_new_pos = np.add(pos, self.action_vectors[action])
        if self.is_valid_position(*candidate_new_pos):
            return self.pos2idx(candidate_new_pos)
        else:
            return state

    def is_valid_position(self, x, y):
        return 0 <= x < self._n_rows and 0 <= y < self._n_cols

    def visualize_rewards(self, r=None, title: str = "Ground-truth rewards", **kwargs):
        """
        Plots a heatmap of rewards (or can be used to plot any reward-like quantity over the grid if passed as and
        argument
        Args:
            r: (Optional) Vector or 2-d array of state rewards. If a 3d array is passed, the expected reward for each state-action
                pair is plotted.
            title: Plot title.
            **kwargs:

        Returns:

        """
        if r is None:
            r = self._r

        if len(r.shape) == 1:
            if len(r)> self._n_cols * self._n_rows:
                r = r[:-1]
            reward_mat = np.reshape(r, (self._n_rows, self._n_cols))
        elif self.reward_type == RewardType.s_source:
            reward_mat = np.reshape(r[:-1, 0, 0], (self._n_rows, self._n_cols))
        elif len(r.shape) == 3:
            # Calculate expected reward for each state-action pair
            er_sa = np.sum(self._p * r, axis=2)
            self.q_heatmap(q=er_sa, title="s-a pair expected rewards")
            return

        img_utils.heatmap2d(reward_mat, title, block=False, **kwargs)

    def visualise_value_function(self):
        if self._opt_values is None:
            self.calculate_value_function()
        value_mat = np.reshape(self._opt_values[:-1], (self._n_rows, self._n_cols))
        img_utils.heatmap2d(value_mat, "Optimal value function")

    def visualize_state(self, s=None):
        """
        Prints the gridworld state as a binary grid (0 everywhere except the current state).
        """
        if s is None:
            s = self._state

        if s == self.n_states - 1:
            print("[Terminal state] (visualization)")
            return

        vis_grid = np.zeros(self.shape)
        vis_grid[self.idx2pos(s)] = 1
        print(vis_grid)

    def render(self, mode='human'):
        self.visualize_state()

    @property
    def shape(self):
        return self._n_rows, self._n_cols

    def reset(self, *, seed=None, options=None):
        self._last_pos = (None, None)
        return super().reset(seed=seed, options=options)

    def q_heatmap(self, q=None, q_function=None, title="Q-values", **kwargs):
        """
        Visualizes the Q-function (or another state-action quantity) as a heatmap over the gridworld (crosses of values
        for each action at each state).
        :param q: (Optional) Q-values for each state-action pair. If not provided, the Q-values are calculated from the
                   reward.
        :param q_function: (Optional) Function that takes a state and returns the Q-values for all actions.
        :return:
        """
        if q is None and q_function is None:
            if self._q is None:
                self.calculate_value_function()
            q = self._q

        elif q is None and q_function is not None:
            q = np.array([np.squeeze(np.array(q_function(np.array([self.get_obs(s)]))[0])) for s in self._states])

        # Each state is represented as a 3x3 grid so that a cross of values for each action can be displayed.
        grid = np.ones((self._n_rows*3, self._n_cols*3)) * np.nan

        for i in range(self._n_rows):
            for j in range(self._n_cols):
                idx = i * self._n_cols + j
                centre_coords = 3*i+1, 3*j+1
                for qval, action_vector in zip(q[idx,:], self.action_vectors):
                    grid[centre_coords[0]+action_vector[0], centre_coords[1]+action_vector[1]] = qval

        img_utils.heatmap2d(grid, title, **kwargs)

    @property
    def height(self):
        return self._n_rows

    @property
    def width(self):
        return self._n_cols


@configurable_factory
def create_obstacle_gridworld(env_config: ObstacleGridworldConfig) -> Callable[[],GridWorld]:

    assert env_config.num_actions == 5


    def env_factory():
        gw_grid = [[-env_config.time_penalty] * env_config.width for h in range(env_config.height)]

        gw_grid[0][0] = f'{-env_config.time_penalty}I'
        if env_config.extra_initial_state:
            gw_grid[env_config.height-2][env_config.width - 1] = f'{-env_config.time_penalty}I'

        gw_grid[0][env_config.width - 1] = f'{env_config.goal_reward}T'

        w_middle = env_config.width // 2
        for h in range(env_config.height // 2):
            gw_grid[h][w_middle] = env_config.obstacle_reward

        env = GridWorld(gw_grid, observations=env_config.observation_type, reward_type=env_config.reward_type,
                        direct_access=env_config.direct_access, epsilon=env_config.epsilon,
                        max_steps=env_config.max_steps)
        return env

    return env_factory


class MuddyGridWorld(GridWorld):
    # Subclass of GridWorld that stores the grid_labels - [1,0,0]=normal, [0,1,0]=muddy, [0,0,1]=reward
    def __init__(self, grid, grid_labels = None, epsilon: float = 0, s0_grid=None, observations=ObservationType.index, reward_type=RewardType.s_target, kernel_grid_size=(3, 3), delayed_terminal=True, **kwargs):
        super().__init__(grid, epsilon, s0_grid, observations, reward_type, kernel_grid_size, delayed_terminal, **kwargs)

        self.grid_labels = [item for sublist in grid_labels for item in sublist]
        self.grid_labels.append([1.,0.,0.]) #final state
        self.grid_labels = [np.argmax(i) for i in self.grid_labels]


if __name__ == "__main__":
    gw_grid = [
        [   0,  0,  0,  0,  '10T'   ],
        [   0,  0,  0,  0,  0       ],
        [   0,  0,  0,  0,  0       ],
        [  '0I',0, '0I',0,  0       ]
    ]

    gw = GridWorld(gw_grid, observations=ObservationType.kernel)
    gw.reset()

    print(gw.step(3))
    print(gw.step(3))
    print(gw.step(3))

    print(gw.step(2))
    print(gw.step(2))
    print(gw.step(2))
    print(gw.step(2))
    print(gw.step(3))
    print(gw.step(2))

    gw.visualize_rewards()
    gw.visualise_value_function()
