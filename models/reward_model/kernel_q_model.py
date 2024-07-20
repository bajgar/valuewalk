import torch


"""
Simple model for approximating the Q function consisting of
(1) a set of inducing points C
(2) A set of parameters theta_q_af that parameterise a Q function by defining the value associated with each inducing point
(3) A kernel function k that defines the similarity between two points in the input space
"""


def kernel_q_model(s_if, theta_q_sa, X_Q_sf, k, n_a=None):
    """
    :param s_if: Set of states on which the Q function is evaluated
    :param a_if: Set of actions on which the Q function is evaluated (same length as s)
    :param theta_q_sa: Parameters of the Q function representing the value of each inducing point and action
    :param X_Q_sf: Set of inducing points for the Q function
    :param k: kernel function
    :return:
    """

    if n_a is not None:
        theta_q_sa = theta_q_sa.reshape(torch.Size([-1, n_a]))

    # Compute the kernel matrix between the inducing points and the input states
    K_is = k(s_if, X_Q_sf)

    # Compute the Q function
    Q_isa = K_is[..., None] * theta_q_sa[None, ...]
    Q_ia = Q_isa.sum(dim=-2)

    return Q_ia
