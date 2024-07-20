import torch
import gymnasium as gym

from models.reward_models.reward_model import RewardModel, MonteCarloRewardModel


class LinearStateRewardModel(torch.nn.Module, RewardModel):
    def __init__(self, input_space: gym.Space, coeffs: torch.Tensor = None):

        torch.nn.Module.__init__(self)

        if coeffs is None:
            self.linear_param = torch.nn.Parameter(torch.randn(input_space.shape[0]))

        else:
            self.linear_param = torch.nn.Parameter(coeffs)

    def point_estimate(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.linear_param @ s

    def save(self, path: str):

        metadata = {
            "reward_model_class": self.__class__.__name__,
            "observation_space": self.observation_space
        }

        torch.save({
            "metadata": metadata,
            "param_samples": self.linear_param
        }, path)

    @classmethod
    def load(cls, path: str):
        checkpoint = torch.load(path)

        metadata = checkpoint["metadata"]
        param_samples = checkpoint["param_samples"]

        assert metadata["reward_model_class"] == cls.__name__

        return cls(metadata["observation_space"], param_samples)


class MonteCarloLinearRewardModel(MonteCarloRewardModel):

    def point_estimate(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.param_expectation @ s
