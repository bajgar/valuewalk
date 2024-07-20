from typing import List

import torch
import torch.nn as nn

from configuration.configurable_factory import configurable_factory


def linear_model(x: torch.Tensor, theta: torch.Tensor, add_bias_ones=True) -> torch.Tensor:
    """

    :param x: a vector with arbitrarily many batch dimensions and a last dimension of size n
    :param theta: an m x (n+1) matrix (if add_bias_ones=True) or an m x n matrix (if add_bias_ones=False)
    :param add_bias_ones: whether to add a column of ones to x
    :return:
    """

    if add_bias_ones:
        x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)

    # Broadcasted matrix multiplication
    output = torch.tensordot(x, theta.movedim(-1, 0), dims=1)

    return output


def mlp_factory(input_size: int, hidden_sizes: list, output_size: int, activation=torch.relu):
    """

    :param input_size: the size of the input
    :param hidden_sizes: a list of hidden sizes
    :param output_size: the size of the output
    :param activation: the activation function to use
    :return: a callable that accepts an input tensor and returns an output tensor
    """

    def mlp(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        :param x: a vector with arbitrarily many batch dimensions and a last dimension of size input_size
        :param theta: a vector of arbitrary batch dimensions and a last dimension of size corresponding to all
            parameters in the MLP
        :return: a vector with arbitrarily many batch dimensions and a last dimension of size output_size
        """
        theta_batch_dims = theta.shape[:-1]
        input_batch_dims = x.shape[:-1]

        if len(theta.shape) > 1:
            # Singleton batch dimensions to x to match theta
            x = x.reshape(*x.shape[:-2], *([1] * len(theta_batch_dims)), *x.shape[-2:])

        thetas = torch.split(theta,
            [(input_size+1) * hidden_sizes[0]] + [(hidden_sizes[i]+1) * hidden_sizes[i + 1]
             for i in range(len(hidden_sizes) - 1)] + [(hidden_sizes[-1]+1) * output_size],
                             dim=-1)

        x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        x = activation(torch.matmul(x, thetas[0].reshape(*theta_batch_dims, input_size + 1, hidden_sizes[0])))
        x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        for i in range(len(hidden_sizes)-1):
            x = activation(torch.matmul(x, thetas[i+1].reshape(*theta_batch_dims, hidden_sizes[i] + 1, hidden_sizes[i+1])))
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        x = torch.matmul(x, thetas[-1].reshape(*theta_batch_dims, hidden_sizes[-1] + 1, output_size))
        if len(theta_batch_dims) > 0 and len(input_batch_dims) > 0:
            x = x.movedim(-2, -len(theta.shape) - 1)
        elif len(input_batch_dims) == 0 and len(theta_batch_dims) > 0:
            x = x.squeeze(-2)
        return x

    return mlp


@configurable_factory
def mlp_q_model_factory(irl_config: 'IRLConfig'):
    """

    :param irl_config: an IRLConfig object
    :return: a callable that accepts an input tensor and returns an output tensor
    """

    return mlp_factory(irl_config.q_model_inputs, irl_config.q_model_hidden_layer_sizes, 1)


def get_num_mlp_params(input_size: int, hidden_sizes: list, output_size: int) -> int:
    """
    Calculates the total number of parameters in an MLP with the given architecture.
    :param input_size: the size of the input
    :param hidden_sizes: a list of hidden sizes
    :param output_size: the size of the output
    :return: the number of parameters in the MLP
    """

    return sum([(input_size + 1) * hidden_sizes[0]] + [(hidden_sizes[i] + 1) * hidden_sizes[i + 1] for i in range(len(hidden_sizes) - 1)] + [(hidden_sizes[-1] + 1) * output_size])


class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: int | List[int], output_dim: int):
        super(MLP, self).__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        if hidden_dims is None or len(hidden_dims) == 0:
            self.layers = nn.Linear(input_dim, output_dim)
        else:
            self.layers = nn.Sequential(
                *([nn.Linear(input_dim, hidden_dims[0]),
                    nn.ELU()] +
                  [nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims) - 1)] +
                  [nn.Linear(hidden_dims[-1], output_dim)])
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
