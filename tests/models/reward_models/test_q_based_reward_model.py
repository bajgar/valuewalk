import pytest
import torch
import gymnasium as gym
from botorch.models import HeteroskedasticSingleTaskGP
from gymnasium.spaces import Box

from models.basic_models import linear_model
from irl_algorithms.demonstrations import Trajectory, Demonstrations
from models.reward_models.q_based_reward_model import QBasedSampleBasedRewardModel

state_dims = 4
action_dims = 2
input_space = Box(low=-1.0, high=1.0, shape=(state_dims+action_dims+1,))


def test_q_samples():

    num_param_samples = 111

    # Create mock data for testing
    theta_q_cf = 2. + 0.02 * torch.randn(num_param_samples, input_space.shape[0])

    # Initialize the QBasedSampleBasedRewardModel
    model = QBasedSampleBasedRewardModel(q_param_samples={'theta_q': theta_q_cf},
                                         q_model=linear_model)

    # Test the point_estimate method
    s_bf = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.,3., 4., 5.]])
    a_bf = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    model.q_model = linear_model
    q_bc = model.q_samples(s_bf, a_bf)
    q_mean_b = torch.mean(q_bc, dim=1)
    target = 2*(torch.sum(s_bf, dim=-1) + torch.sum(a_bf, dim=-1) + 1.0)

    assert q_bc.shape == (2, num_param_samples), "q_bc shape mismatch"
    assert torch.allclose(q_mean_b, target, rtol=0.01)


#############################
### TEST GP INTERPOLATION ###
#############################

# Create mock data for testing
def create_mock_data(num_samples=100, state_dim=4, action_dim=2, num_posterior_samples=77):
    states = torch.randn(num_samples, state_dim)
    actions = torch.randn(num_samples, action_dim)
    rewards = ((torch.sin(states).sum(dim=1, keepdim=True) + torch.cos(actions).sum(dim=1, keepdim=True)) *
               torch.randn(1, num_posterior_samples))
    reward_var = torch.var(rewards, dim=1)
    return states, actions, rewards, reward_var


@pytest.fixture
def reward_model():
    num_posterior_samples = 77
    states, actions, rewards, reward_var = create_mock_data(num_posterior_samples=num_posterior_samples)
    q_param_samples = {'theta_q': torch.randn(num_posterior_samples, states.shape[1] + actions.shape[1])}
    model = QBasedSampleBasedRewardModel(
        input_space=input_space,
        q_param_samples=q_param_samples,
        q_model=linear_model,
        reward_eval_states=states,
        reward_eval_actions=actions,
        reward_samples=rewards
    )
    return model
