import torch
from models.reward_models.variational_reward_model import NormalVariationalRewardModel


def test_point_estimate():
    """
    Single continuous action
    """
    reward_model = torch.nn.Linear(3, 2)
    model = NormalVariationalRewardModel(reward_model)
    s = torch.tensor([1.0, 2.0])
    a = torch.tensor([1.0])
    assert model.point_estimate(s, a).shape == torch.Size([])


def test_point_estimate_batch():
    """
    Batch of discrete actions
    """
    num_actions = 3
    batch_size = 2

    reward_model = torch.nn.Linear(3, 2*num_actions)
    model = NormalVariationalRewardModel(reward_model)
    s = torch.tensor([[1., 2., 2.], [3., 4., 4.]])
    a = torch.tensor([[2], [0]])
    assert model.point_estimate(s, a).shape == torch.Size([batch_size])


def test_mean():
    """
    Single continuous action
    """
    num_actions = 2

    reward_model = torch.nn.Linear(2, 2*num_actions)
    model = NormalVariationalRewardModel(reward_model)
    s = torch.tensor([1.0, 2.0])
    a = torch.tensor([1])
    assert model.mean(s, a).shape == torch.Size([])


def test_std():
    reward_model = torch.nn.Linear(3, 2)
    model = NormalVariationalRewardModel(reward_model)
    s = torch.tensor([1.0, 2.0])
    a = torch.tensor([1.0])
    assert model.std(s, a).shape == torch.Size([])


def test_std_batch():
    reward_model = torch.nn.Linear(4, 2)
    model = NormalVariationalRewardModel(reward_model)
    s = torch.tensor([[1.0, 2.0, 5.], [3., 4., 6.]])
    a = torch.tensor([[1.0], [2.]])
    assert model.std(s, a).shape == torch.Size([2])


def test_variance():
    reward_model = torch.nn.Linear(3, 2)
    model = NormalVariationalRewardModel(reward_model)
    s = torch.tensor([1.0, 2.0])
    a = torch.tensor([1.0])
    assert model.variance(s, a).shape == torch.Size([])


def test_quantile():
    reward_model = torch.nn.Linear(3, 2)
    model = NormalVariationalRewardModel(reward_model)
    s = torch.tensor([1.0, 2.0])
    a = torch.tensor([1.0])
    q = 0.5
    assert model.quantile(s, a, q).shape == torch.Size([])
