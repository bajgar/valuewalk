from pathlib import Path
from typing import Callable

import pydantic
from ray.rllib.algorithms import AlgorithmConfig

from irl_algorithms.utils.collect_trajectories import collect_boltzmann_demos_factory_finite, collect_demos_from_loaded_policy


class DemosConfig(pydantic.BaseModel):
    demo_factory: Callable
    demos_id: str = None
    valid_demo_factory: Callable = None
    n_trajectories: int = None
    total_samples: int = None
    append_last: bool = True

    expert_policy_factory: Callable = None
    rollout_function: Callable = None

    demos_file_path: str | Path = None

    save_path: str | Path = None

    demo_subset_randomization: bool = False  # If more than n_trajectories are available, select randomly.

    onehot_states_to_index: bool = False
    index_to_onehot: bool = False
    append_last_dummy: bool = False

    data_split: str = "train"


    @pydantic.validator("total_samples")
    def check_total_samples(cls, v, values):
        if v is None and values.get("n_trajectories") is None:
            raise ValueError("Either n_trajectories or total_samples must be specified")
        return v


class BoltzmannDemosConfig(DemosConfig):
    demo_factory: Callable = pydantic.Field(default_factory=lambda: collect_boltzmann_demos_factory_finite)
    beta_expert: float = 1.
    gamma: float = 0.9
    seed: int = None
    env_name: str = None
    num_action_samples: int = 20  # Number of actions to sample for Boltzmann policy in continuous environments



class DemosFromRLCheckpointConfig(DemosConfig):

    demo_factory: Callable = collect_demos_from_loaded_policy
    rl_checkpoint: str | Path
    rl_config_factory: Callable[[], dict | AlgorithmConfig]

    class Config:
        arbitrary_types_allowed = True


class BoltzmannDemosFromRLCheckpointConfig(DemosFromRLCheckpointConfig, BoltzmannDemosConfig):
    pass
