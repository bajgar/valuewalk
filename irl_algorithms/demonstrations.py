from typing import List, Callable

import pydantic
import torch
from pydantic import dataclasses


@dataclasses.dataclass
class Trajectory:
    oa_pairs: List[tuple] = pydantic.Field(default_factory=list)

    truncated: bool = None
    terminated: bool = None

    def __len__(self):
        return len(self.oa_pairs)

    def __getitem__(self, idx):
        return self.oa_pairs[idx]

    def __iter__(self):
        return iter(self.oa_pairs)

    def __add__(self, other: 'Trajectory'):
        return Trajectory(self.oa_pairs + other.oa_pairs)

    def __radd__(self, other: 'Trajectory'):
        return Trajectory(other.oa_pairs + self.oa_pairs)

    def append(self, s: torch.Tensor, a: torch.Tensor) -> None:
        # check type
        if not isinstance(s, torch.Tensor):
            raise TypeError(f"Expected state to be a torch.Tensor, got {type(s)}")
        if not isinstance(a, torch.Tensor):
            raise TypeError(f"Expected action to be a torch.Tensor, got {type(a)}")
        self.oa_pairs.append((s, a))

    @property
    def states(self) -> List[torch.Tensor]:
        state_list = [s for s, _ in self.oa_pairs]
        return state_list

    @property
    def states_tensor(self) -> torch.Tensor:
        state_tensor = torch.stack(self.states)
        return state_tensor

    @property
    def actions(self) -> List[torch.Tensor]:
        return [a for _, a in self.oa_pairs]

    @property
    def actions_tensor(self) -> torch.Tensor:
        action_tensor = torch.stack(self.actions)
        return action_tensor

    def feature_sum(self, onehotify: bool = False, n_states: int = None) -> torch.Tensor:
        """
        Returns the sum of the features of the states in the trajectory.
        :param onehotify: convert integer states into 1-hot vectors
        :param n_states: length of 1-hot vectors if onehotify is True
        :return:
        """
        states = self.states_tensor
        if onehotify:
            states = torch.nn.functional.one_hot(states, n_states)
        return torch.sum(states, dim=0)

    def get_two_step_pairs(self):
        """
        Returns two tensors: on of pairs of consecutive observations and one of pairs of consecutive actions
        """
        states = self.states_tensor
        actions = self.actions_tensor
        return torch.stack([states[:-1], states[1:]], dim=1), torch.stack([actions[:-1], actions[1:]], dim=1)

    def get_oa_tensor(self):
        return torch.stack([torch.cat([s, a]) for s, a in self.oa_pairs])


@dataclasses.dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Demonstrations:
    trajectories: List[Trajectory] = pydantic.Field(default_factory=list)

    cache_oa_tensor: bool = True
    _oa_tensor_no_ep_ends: torch.Tensor = None
    _oa_tensor_ep_ends: torch.Tensor = None

    def get_states(self, omit_episode_end=False):
        if omit_episode_end:
            return [s for trajectory in self.trajectories for s in trajectory.states[:-1]]
        return [s for trajectory in self.trajectories for s in trajectory.states]

    @property
    def states(self):
        return self.get_states()

    def get_states_tensor(self, omit_episode_end=False):
        return torch.stack(self.get_states(omit_episode_end=omit_episode_end))

    @property
    def states_tensor(self):
        return self.get_states_tensor()

    def get_actions(self, omit_episode_end=False):
        if omit_episode_end:
            return [a for trajectory in self.trajectories for a in trajectory.actions[:-1]]
        return [a for trajectory in self.trajectories for a in trajectory.actions]

    @property
    def actions(self):
        return self.get_actions()

    def get_actions_tensor(self, omit_episode_end=False):
        return torch.stack(self.get_actions(omit_episode_end=omit_episode_end))

    @property
    def actions_tensor(self):
        return self.get_actions_tensor()

    def get_oa_pairs(self, omit_episode_end=False) -> List[tuple]:
        """
        Returns a single list of (observation, action) pairs across all trajectories.
        :param omit_episode_end: Omits the last observation-action pair of each trajectory.
        :return:
        """
        if omit_episode_end:
            return [oa_pair for trajectory in self.trajectories for oa_pair in trajectory[:-1]]
        return [oa_pair for trajectory in self.trajectories for oa_pair in trajectory]

    @property
    def oa_pairs(self, omit_episode_end=False) -> List[tuple]:
        return self.get_oa_pairs(omit_episode_end=omit_episode_end)

    def get_oa_tensor(self, omit_episode_end=False) -> torch.Tensor:
        """
        Returns an N x [observation_dim + action_dim] tensor where each row is a concatenated observation-action pair.
        :param omit_episode_end: Omits the last observation-action pair of each trajectory.
        :return:
        """

        if omit_episode_end:
            if self._oa_tensor_no_ep_ends is None:
                self._oa_tensor_no_ep_ends = torch.stack([torch.cat([s, a])
                                                          for s, a in self.get_oa_pairs(omit_episode_end=omit_episode_end)])
            return self._oa_tensor_no_ep_ends

        else:
            if self._oa_tensor_ep_ends is None:
                self._oa_tensor_ep_ends = torch.stack([torch.cat([s, a])
                                                       for s, a in self.get_oa_pairs(omit_episode_end=omit_episode_end)])
            return self._oa_tensor_ep_ends

    def average_feature_sum(self, onehotify=False):
        if onehotify:
            n_states = torch.max(self.states_tensor) + 1

        return torch.mean(torch.stack([
            trajectory.feature_sum(onehotify=onehotify, n_states=n_states) for trajectory in self.trajectories]), dim=0)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Demonstrations(trajectories=self.trajectories[idx])
        else:
            return self.trajectories[idx]

    def append(self, trajectory: Trajectory):
        # invalidate cache
        self._oa_tensor_no_ep_ends = None
        self._oa_tensor_ep_ends = None

        self.trajectories.append(trajectory)

    def extend(self, other_demonstrations: 'Demonstrations'):
        # invalidate cache
        self._oa_tensor_no_ep_ends = None
        self._oa_tensor_ep_ends = None

        assert isinstance(other_demonstrations, Demonstrations)
        self.trajectories.extend(other_demonstrations.trajectories)

    def sa_counts(self, n_actions: int = None, onehot: bool = True, n_states: int = None):
        """
        If onehot==True, assumes the observations are 1-hot encoded and actions are integers. Else, assumes both are
        integers.
        :param onehot: whether the observations are 1-hot encoded
        :param n_actions: number of actions (if None, inferred as the max action + 1)
        :return: counts of each state-action pair
        """
        states = self.states_tensor
        if onehot:
            assert n_states is None or n_states == states.shape[1]
            n_states = states.shape[1]
            states_1dxs = states.argmax(dim=1)
        else:
            if n_states is None:
                n_states = int(states.max() + 1)
            states_1dxs = states
        if n_actions is None:
            n_actions = self.actions_tensor.max() + 1
        counts = torch.zeros(n_states, n_actions)
        for s, a in zip(states_1dxs, self.actions_tensor):
            counts[int(s), int(a)] += 1
        return counts

    def __len__(self):
        return len(self.trajectories)

    def save(self, path):
        states_tensors = [trajectory.states_tensor for trajectory in self.trajectories]
        actions_tensors = [trajectory.actions_tensor for trajectory in self.trajectories]
        terminated = [trajectory.terminated for trajectory in self.trajectories]
        truncated = [trajectory.truncated for trajectory in self.trajectories]
        torch.save((states_tensors, actions_tensors, terminated, truncated), path)

    @classmethod
    def load(cls, path):
        states_tensors, actions_tensors, terminateds, truncateds = torch.load(path)
        demonstrations = []
        for states_tensor, actions_tensor, terminated, truncated in zip(states_tensors, actions_tensors, terminateds, truncateds):
            trajectory = Trajectory()
            for s, a in zip(states_tensor, actions_tensor):
                trajectory.append(s, a)
            trajectory.terminated = terminated
            trajectory.truncated = truncated
            demonstrations.append(trajectory)
        return Demonstrations(demonstrations)

    def get_two_step_pairs(self):
        """
        Returns two tensors: on of pairs of consecutive observations and one of pairs of consecutive actions
        """
        state_pairs = []
        action_pairs = []
        for traj in self.trajectories:
            s, a = traj.get_two_step_pairs()
            state_pairs.append(s)
            action_pairs.append(a)

        return torch.cat(state_pairs, dim=0), torch.cat(action_pairs, dim=0)

