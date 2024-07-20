from pathlib import Path
from typing import Callable, Union, List, Dict, Tuple, Optional
from abc import ABC, abstractmethod

import gymnasium as gym
import pydantic
import torch

from irl_algorithms.demonstrations import Demonstrations
from models.basic_models import get_num_mlp_params
from models.reward_models.reward_model import RewardModel


SizeType = Union[torch.Size, List[int], Tuple[int]]


class IRLConfig(pydantic.BaseModel):
    irl_method_factory: Callable[['gym.Env', 'IRLConfig'], 'IRLMethod']
    num_params: int = None

    gamma: float = 0.9
    verbose: bool = False
    weight_decay: float = 0.0
    zero_final_reward: bool = True  # For environments with dummy final post-termination state

    q_model: Callable[['IRLConfig'], Union[Callable, torch.nn.Module]] = None
    q_model_inputs: int = None
    q_model_hidden_layer_sizes: Optional[List[int]] = None

    embedding_dim: int = 4

    preprocessing_module_factory: Optional[Callable[['IRLConfig'], torch.nn.Module]] = None

    beta_expert: float = 4.0

    approximation_grid_size: tuple = (3, 3)
    num_synth_trajectories: int = 0
    max_synth_traj_length: int = 100
    num_action_samples: int = None

    epochs: int = 100
    batch_size: int = 64
    aux_demo_factory: Callable[[Demonstrations, gym.Env], Demonstrations] | Callable[['IRLConfig'], Callable[[Demonstrations, gym.Env], Demonstrations]] = None

    last_state_q_penalty: float = 0.0  # For environments with terminal states, this is the penalty for the
                                        # Q value of the last state in the trajectory (used in AVRIL)
    compute_KL_using_all_states: bool = True
    compute_TDL_using_all_states: bool = True

    # Note that checkpointing needs to be handled by the IRLMethod, else these params have no effect
    checkpoint_frequency: Optional[int] = 1000
    checkpoint_path: str | Path = None  # If None, will be set to result_save_path.with_suffix(".checkpoint.pt")

    test_episodes: int = 100

    @property
    def q_model_params(self):
        if self.q_model_hidden_layer_sizes is None:
            return self.q_model_inputs + 1
        else:
            return get_num_mlp_params(self.q_model_inputs, self.q_model_hidden_layer_sizes, 1)


class IRLMethod(ABC):
    def __init__(self, env: gym.Env, config: IRLConfig):
        self.env = env
        self.config = config

    @abstractmethod
    def run(self, trajectories_t: Demonstrations) -> RewardModel:
        raise NotImplementedError
