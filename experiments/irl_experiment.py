import pathlib
import datetime
import logging
from typing import Optional, Dict

import torch

from configuration.configurable_factory import resolve_configurable_factories
from envs.env_config import EnvConfig
from envs.gridworld import ObservationType
from experiments.experiment_config import ExperimentConfig
from irl_algorithms.demonstrations_config import DemosConfig
from irl_algorithms.irl_method import IRLConfig
from models.reward_models.reward_model import RewardModel


logging.getLogger().setLevel(logging.INFO)


class IRLExperimentConfig(ExperimentConfig):
    irl_config: IRLConfig
    env_config: EnvConfig
    demos_config: DemosConfig

    result_save_path: Optional[pathlib.Path]
    save_method: str = "torch"


class IRLExperiment:

    def __init__(self, config: IRLExperimentConfig):
        self.config = config

    def resolve_configs(self):

        if self.config.irl_config.checkpoint_path is None:
            self.config.irl_config.checkpoint_path = self.config.result_save_path.with_suffix(".checkpoint.pt")

        resolve_configurable_factories(self.config.env_config)
        resolve_configurable_factories(self.config.irl_config)
        resolve_configurable_factories(self.config.demos_config)

    def run(self) -> (RewardModel, Dict):
        start_time = datetime.datetime.now()
        logging.info(f"Starting IRLExperiment run at {start_time}.")

        self.resolve_configs()

        env = self.config.env_config.env_factory()
        D = self.config.demos_config.demo_factory(env)

        logging.info(f"Prepared environment and demonstrations in {datetime.datetime.now() - start_time}. "
                     f"Starting IRL run.")


        if self.config.demos_config.valid_demo_factory is not None:
            D_valid = self.config.demos_config.valid_demo_factory(env)
        else:
            D_valid = None

        logging.info(f"Prepared environment and demonstrations in {datetime.datetime.now() - start_time}. "
                     f"Starting IRL run.")

        irl_method = self.config.irl_config.irl_method_factory(env, self.config.irl_config)
        if D_valid is None:
            reward_model, info = irl_method.run(D)
        else:
            reward_model, info = irl_method.run(D, D_valid)

        logging.info(f"Finished IRL run in {datetime.datetime.now() - start_time}.")
        info["demonstrations"] = D
        info["total_time"] = datetime.datetime.now() - start_time
        self.save_results(reward_model, info)

        return reward_model, info

    def save_results(self, reward_model: RewardModel, info: Dict):
        if self.config.result_save_path is None:
            return

        logging.info(f"Saving reward model to {self.config.result_save_path}.")

        if hasattr(reward_model, "save"):
            reward_model.save(self.config.result_save_path)
        elif self.config.save_method == "torch":
            torch.save(reward_model, self.config.result_save_path)
        elif self.config.save_method == "pickle":
            import pickle
            with open(self.config.result_save_path, "wb") as f:
                pickle.dump(reward_model, f)
        else:
            raise ValueError(f"Unknown save method {self.config.save_method}.")

        if info:
            info_save_path = self.config.result_save_path.with_suffix(".info.pt")
            torch.save(info, info_save_path)
