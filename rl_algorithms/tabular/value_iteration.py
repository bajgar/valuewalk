from typing import Optional

import torch
import numpy as np


class ValueIteration:
    """
    Value iteration algorithm for solving MDPs, which can cache the previous value vector for a speedup if run
    repeatedly on an evolving MDP. This work for 1-D or 2-D rewards depending on the source state (not target state).
    """

    def __init__(self):
        self._last_q = None

    def run(self, P_sas: torch.Tensor, r: torch.Tensor,
            q_prev: Optional[torch.Tensor] = None,
            gamma: float =0.9, max_iters: int = 1000, tol: float = 1e-4) -> torch.Tensor:
        """
        :param P_sas: [n_states] x [n_actions] x [n_states] tensor where p[s0, a, s1] is the probability of
                    transitioning from s0 to s1 if action a is taken
        :param r: reward vector
        :param q_prev: previous Q values to start from
        :param gamma: discount rate
        :param max_iters: maximum number of iterations
        :param tol: tolerance for convergence

        :return: Q values
        """

        n_states, n_actions, _ = P_sas.shape
        if q_prev is not None:
            Q_sa = q_prev
            self._last_q = q_prev
        elif self._last_q is not None:
            Q_sa = self._last_q
        else:
            Q_sa = torch.zeros([n_states, n_actions])
            self._last_q = Q_sa.detach()

        if len(r.shape) == 1:
            r = r.unsqueeze(1)

        for it in range(max_iters):
            # Bellman update
            V_s = Q_sa.max(dim=1)[0]
            Q_sa = r + gamma * P_sas @ V_s

            # Test for convergence
            q_diff = torch.max(torch.abs(Q_sa - self._last_q))
            if q_diff < tol:
                break
            self._last_q = Q_sa.detach()

        return Q_sa


class ValueIterationNP:
    """Same as ValueIteration but for numpy arrays"""

    def __init__(self):
        self._last_q = None

    def run(self, P_sas, r,
            q_prev=None,
            gamma=0.9, max_iters=1000, tol=1e-4):
        """
        :param P_sas: [n_states] x [n_actions] x [n_states] tensor where p[s0, a, s1] is the probability of
                    transitioning from s0 to s1 if action a is taken
        :param r: reward vector
        :param q_prev: previous Q values to start from
        :param gamma: discount rate
        :param max_iters: maximum number of iterations
        :param tol: tolerance for convergence

        :return: Q values
        """

        n_states, n_actions, _ = P_sas.shape
        if q_prev is not None:
            Q_sa = q_prev
            self._last_q = q_prev
        elif self._last_q is not None:
            Q_sa = self._last_q
        else:
            Q_sa = np.zeros([n_states, n_actions])
            self._last_q = Q_sa

        if len(r.shape) == 1:
            r = r.reshape(-1, 1)

        for it in range(max_iters):
            # Bellman update
            V_s = Q_sa.max(axis=1)
            Q_sa = r + gamma * np.matmul(P_sas, V_s)

            # Test for convergence
            q_diff = np.max(np.abs(Q_sa - self._last_q))
            if q_diff < tol:
                break
            self._last_q = Q_sa

        return Q_sa