from __future__ import annotations

from typing import Sequence
import numpy as np
import jax.numpy as jnp
import jax.nn as jnn
from flax import linen as nn

from configuration.configurable_factory import configurable_factory


def linear_model(
    x: jnp.ndarray,
    theta: jnp.ndarray,
    add_bias_ones: bool = True,
) -> jnp.ndarray:
    
    """
    :param x: a vector with arbitrarily many batch dimensions and a last dimension of size n
    :param theta: an m x (n+1) matrix (if add_bias_ones=True) or an m x n matrix (if add_bias_ones=False)
    :param add_bias_ones: whether to add a column of ones to x
    :return:
    """
    
    if add_bias_ones:
        x = jnp.concatenate([x, jnp.ones_like(x[..., :1])], axis=-1)

    return jnp.tensordot(x, jnp.moveaxis(theta, -1, 0), axes=1)


def mlp_factory_jax(
    input_size: int,
    hidden_sizes: Sequence[int],
    output_size: int,
    activation=jnn.relu,
):
    hidden_sizes = tuple(hidden_sizes)
    if len(hidden_sizes) == 0:
        raise ValueError("mlp_factory requires at least one hidden layer.")

    lengths = (
        [(input_size + 1) * hidden_sizes[0]]
        + [(hidden_sizes[i] + 1) * hidden_sizes[i + 1] for i in range(len(hidden_sizes) - 1)]
        + [(hidden_sizes[-1] + 1) * output_size]
    )

    offsets = np.cumsum((0, *lengths))
    slices = tuple(
        slice(int(offsets[i]), int(offsets[i + 1]))
        for i in range(len(lengths))
    )

    def mlp(x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        theta_batch_dims = theta.shape[:-1]
        input_batch_dims = x.shape[:-1]

        if theta.ndim > 1:
            x = x.reshape(*x.shape[-2:], *([1] * len(theta_batch_dims)), *x.shape[-2:])

        thetas = [theta[..., sl] for sl in slices]

        x = jnp.concatenate([x, jnp.ones_like(x[..., :1])], axis=-1)
        x = activation(
            jnp.matmul(
                x,
                thetas[0].reshape(*theta_batch_dims, input_size + 1, hidden_sizes[0]),
            )
        )

        x = jnp.concatenate([x, jnp.ones_like(x[..., :1])], axis=-1)
        for i in range(len(hidden_sizes) - 1):
            x = activation(
                jnp.matmul(
                    x,
                    thetas[i + 1].reshape(
                        *theta_batch_dims,
                        hidden_sizes[i] + 1,
                        hidden_sizes[i + 1],
                    ),
                )
            )
            x = jnp.concatenate([x, jnp.ones_like(x[..., :1])], axis=-1)

        x = jnp.matmul(
            x,
            thetas[-1].reshape(*theta_batch_dims, hidden_sizes[-1] + 1, output_size),
        )

        if len(theta_batch_dims) > 0 and len(input_batch_dims) > 0:
            x = jnp.moveaxis(x, -2, -len(theta.shape) - 1)
        elif len(input_batch_dims) == 0 and len(theta_batch_dims) > 0:
            x = jnp.squeeze(x, axis=-2)

        return x

    return mlp

@configurable_factory
def mlp_q_model_factory_jax(irl_config: 'IRLConfig'):

    return mlp_factory_jax(irl_config.q_model_inputs, irl_config.q_model_hidden_layer_sizes, 1)


def get_num_mlp_params(input_size: int, hidden_sizes: Sequence[int], output_size: int) -> int:
    hidden_sizes = list(hidden_sizes)
    return sum(
        [(input_size + 1) * hidden_sizes[0]]
        + [(hidden_sizes[i] + 1) * hidden_sizes[i + 1] for i in range(len(hidden_sizes) - 1)]
        + [(hidden_sizes[-1] + 1) * output_size]
    )


class MLP(nn.Module):
    input_dim: int
    hidden_dims: int | Sequence[int] | None
    output_dim: int
    activation: callable = nn.elu
    
    def setup(self):
        if isinstance(self.hidden_dims, int):
            hidden_dims = [self.hidden_dims]
        elif self.hidden_dims is None:
            hidden_dims = []
        else:
            hidden_dims = list(self.hidden_dims)

        self.hidden_layers = [nn.Dense(h) for h in hidden_dims]
        self.output_layer = nn.Dense(self.output_dim)

    def __call__(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)