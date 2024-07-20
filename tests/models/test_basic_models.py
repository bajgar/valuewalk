import pytest
import torch

from configuration.configurable_factory import resolve_configurable_factories
from irl_algorithms.irl_method import IRLConfig
from models.basic_models import linear_model, mlp_factory, mlp_q_model_factory, get_num_mlp_params

# Batch dims for the 2d and 3d tensors
batch_dim_options = [(), (5,), (5, 6)]
input_dim_options = [1, 3]
output_dim_options = [1, 3]
bias_options = [True, False]
theta_batch_dims = [(), (4,), (4, 7)]


# Apply Cartesian product using multiple parametrize decorators
@pytest.mark.parametrize("batch_dims", batch_dim_options)
@pytest.mark.parametrize("input_features", input_dim_options)
@pytest.mark.parametrize("output_dims", output_dim_options)
@pytest.mark.parametrize("add_bias_ones", bias_options)
@pytest.mark.parametrize("theta_batch_dims", theta_batch_dims)
def test_linear_model(batch_dims, input_features, output_dims, add_bias_ones, theta_batch_dims):
    theta_dim = input_features + 1 if add_bias_ones else input_features
    theta = torch.randn(theta_batch_dims + (output_dims, theta_dim))

    input_tensor_shape = batch_dims + (input_features,)

    x = torch.randn(input_tensor_shape)
    y = linear_model(x, theta, add_bias_ones=add_bias_ones)

    expected_shape = batch_dims + theta_batch_dims + (output_dims,)
    assert y.shape == expected_shape, f"Expected shape {expected_shape}, but got {y.shape}"


@pytest.mark.parametrize("batch_dims", batch_dim_options)
@pytest.mark.parametrize("input_features", input_dim_options)
@pytest.mark.parametrize("output_dims", output_dim_options)
@pytest.mark.parametrize("hidden_sizes", [[10], [10, 20, 30]])
@pytest.mark.parametrize("theta_batch_dims", theta_batch_dims)
def test_mlp_factory(batch_dims, input_features, output_dims, hidden_sizes, theta_batch_dims):
    theta_size = sum([(input_features + 1) * hidden_sizes[0]] + [(hidden_sizes[i]+1) * hidden_sizes[i + 1] for i in range(len(hidden_sizes) - 1)] + [(hidden_sizes[-1]+1) * output_dims])
    theta = torch.randn(theta_batch_dims + (theta_size,))

    input_tensor_shape = batch_dims + (input_features,)

    x = torch.randn(input_tensor_shape)
    mlp = mlp_factory(input_features, hidden_sizes, output_dims)
    y = mlp(x, theta)

    expected_shape = batch_dims + theta_batch_dims + (output_dims,)
    assert y.shape == expected_shape, f"Expected shape {expected_shape}, but got {y.shape}"


def test_mlp_q_model_factory():

    input_tensor_shape = (5, 6, 3)
    x = torch.randn(input_tensor_shape)

    config = IRLConfig(q_model_inputs=input_tensor_shape[-1], q_model_hidden_layer_sizes=[10, 20], q_model=mlp_q_model_factory,
                       irl_method_factory=lambda env, config: None)
    resolve_configurable_factories(config)
    theta_size = get_num_mlp_params(config.q_model_inputs, config.q_model_hidden_layer_sizes, 1)
    theta = torch.randn(theta_size)

    y = config.q_model(x, theta)

    expected_shape = (5, 6, 1)
    assert y.shape == expected_shape, f"Expected shape {expected_shape}, but got {y.shape}"
