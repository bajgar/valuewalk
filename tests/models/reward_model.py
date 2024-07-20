import pytest
import torch
import gymnasium as gym
from irl_algorithms.demonstrations import Trajectory, Demonstrations
from models.reward_models.q_based_reward_model import QBasedSampleBasedRewardModel


def test_point_estimate():
    # Define a simple q_model for testing
    def q_model(x, params):
        return torch.sum(x, dim=-1, keepdim=True).unsqueeze(1) + params[None, :, :]

    # Initialize the QBasedSampleBasedRewardModel
    model = QBasedSampleBasedRewardModel({'theta_q': torch.tensor([[0.0], [2.0]])},
                                         q_model)
    model.q_model = q_model

    # Test the point_estimate method
    s_bf = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    a_bf = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    # The params average to 1. I.e the result should be sum of all features plus one.
    assert torch.allclose(model.point_estimate(s_bf, a_bf), torch.tensor([15.0, 23.0]))


def test_q_samples():
    # Define a simple q_model for testing
    def q_model(x, params):
        return torch.sum(x, dim=-1, keepdim=True).unsqueeze(1) + params[None, :, :]

    # Initialize the QBasedSampleBasedRewardModel
    model = QBasedSampleBasedRewardModel({'theta_q': torch.tensor([[1.0],[2.0]])},
                                         q_model)
    model.q_model = q_model

    # Test the q_samples method
    s_bf = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    a_bf = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    q_bc = model.q_samples(s_bf, a_bf)

    assert torch.allclose(q_bc, torch.tensor([[15.0, 16.0],[23.0, 24.0]]))
