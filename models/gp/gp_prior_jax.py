from typing import Callable

import gpjax as gpx
from gpjax.kernels import RBF
import jax.numpy as jnp


class GPPriorJax:
    """
    Gaussian Process prior (primarily used for continuous-space reward functions).

    :param mean_function: Mean function of the GP prior.
    :param kernel: Kernel of the GP prior.
    """
    def __init__(self,
                 mean_function: Callable[jnp.ndarray, jnp.ndarray] | None = None,
                 kernel: Callable[jnp.ndarray, jnp.ndarray] | None = None):

        self.mean_function = mean_function or gpx.mean_functions.Zero()

        self.kernel = kernel or RBF()

        self.dist = gpx.gps.Prior(mean_function=self.mean_function, kernel=self.kernel)

    def log_prob(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(self._dist_at(x).log_prob(y=y))

    def _dist_at(self, x: jnp.ndarray):
        if x.ndim==1:
            x=x[:,None]
        return self.dist.predict(x,return_covariance_type='dense')
    

class PrecomputedGPPriorJax:
    """
    Gaussian Process prior (primarily used for continuous-space reward functions).

    :param mean_function: Mean function of the GP prior.
    :param kernel: Kernel of the GP prior.
    """
    def __init__(self, 
                 eval_points: jnp.ndarray | None = None,
                 mean_function=None, 
                 kernel=None):
        self.mean_function = mean_function or gpx.mean_functions.Zero()
        self.kernel = kernel or gpx.kernels.RBF()
        self.prior = gpx.gps.Prior(
            mean_function=self.mean_function,
            kernel=self.kernel,
        )
        self.dist = None

        if eval_points is not None:
            self.precompute(eval_points)

    def precompute(self, x_eval: jnp.ndarray):
        if x_eval.ndim == 1:
            x_eval = x_eval[:, None]
        self.dist = self.prior.predict(x_eval, return_covariance_type="dense")

    def log_prob(self, x: jnp.ndarray, y: jnp.ndarray):
        if self.dist is None:
            raise ValueError("Call precompute(x_eval) before log_prob.")
        return self.dist.log_prob(y)
