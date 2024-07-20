from typing import Dict

import torch
import gymnasium as gym


class RewardModel:

    def point_estimate(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MonteCarloRewardModel(RewardModel):

    def __init__(self, param_samples: torch.Tensor, aux_samples: Dict[str, torch.Tensor] = None):
        self.param_samples = param_samples
        self.aux_samples = aux_samples

    @property
    def param_expectation(self):
        return torch.mean(self.param_samples, dim=0)
