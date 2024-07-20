import numpy as np
import torch


def value_iteration(p: np.array, r: np.array, gamma=0.9, tol=0.01, max_iters=None, verbose=False):
    """
    Numpy-only implementation of value iteration

    :param p: [n_states] x [n_actions] x [n_states] tensor where p[s0, a, s1] is the probability of transitioning from
                s0 to s1 if action a is taken
    :param r: reward vector
    :param gamma: discount rate
    :param tol: tolerance for convergence
    :return: V: state-value vector
    """
    if isinstance(r, torch.Tensor):
        r = r.detach().numpy()
    if isinstance(p, torch.Tensor):
        p = p.detach().numpy()

    n_states, n_actions, _ = np.shape(p)
    V = np.zeros([n_states])        # The state value function

    err = 9.e9
    n_iters = 0

    if verbose:
        print("Starting value iteration ...")

    while err > tol and (max_iters is None or n_iters < max_iters):

        prev_V = V.copy()

        # # Expected return of arriving to each state
        # exp_r = r + gamma*V

        if len(r.shape) == 3:
            Q = np.sum(p * (r + gamma * V[None, None, :]), axis=2)
        else:
            Q = np.matmul(p, V)

        if len(r.shape) == 1:
            V = r + gamma * np.max(Q, axis=1)
        elif len(r.shape) == 2:
            V = np.max(r + gamma * Q, axis=1)
        elif len(r.shape) == 3:
            V = np.max(Q, axis=1)
        else:
            raise ValueError("r must be at most rank 3")


        err = max(np.abs(V-prev_V))

        if verbose and n_iters % 10 == 0:
            print(f"Iteration {n_iters}\t err: {err}")
        n_iters += 1

    if verbose:
        print(f"Finished after {n_iters} iterations with err: {err}")

    return V, Q


def value_iteration_torch(p: torch.Tensor, r: torch.Tensor, gamma=0.9, tol=0.01, max_iters=None, verbose=False):
    """
    :param p: [n_states] x [n_actions] x [n_states] tensor where p[s0, a, s1] is the probability of transitioning from
                s0 to s1 if action a is taken
    :param r: reward vector
    :param gamma: discount rate
    :param tol: tolerance for convergence
    :return: V: state-value vector
    """
    n_states, n_actions, _ = p.shape
    V = torch.zeros([n_states], device=p.device)        # The state value function

    err = 9.e9
    n_iters = 0

    if verbose:
        print("Starting value iteration ...")

    while err > tol and (max_iters is None or n_iters < max_iters):

        prev_V = V.clone()

        if len(r.shape) == 3:
            Q = torch.sum(p * (r + gamma * V[None, None, :]), dim=2)
        else:
            Q = torch.matmul(p, V)

        if len(r.shape) == 1:
            V = r + gamma * torch.max(Q, dim=1)[0]
        elif len(r.shape) == 2:
            V = torch.max(r + gamma * Q, dim=1)[0]
        elif len(r.shape) == 3:
            V = torch.max(Q, dim=1)[0]
        else:
            raise ValueError("r must be at most rank 3")

        err = torch.max(torch.abs(V-prev_V)).item()

        if verbose and n_iters % 10 == 0:
            print(f"Iteration {n_iters}\t err: {err}")
        n_iters += 1

    if verbose:
        print(f"Finished after {n_iters} iterations with err: {err}")

    return V, Q


def value_iteration_s_only_torch(p: torch.Tensor, r: torch.Tensor, gamma=0.9, tol=0.01, max_iters=None, verbose=False):
    """
    :param p: [n_states] x [n_actions] x [n_states] tensor where p[s0, a, s1] is the probability of transitioning from
                s0 to s1 if action a is taken
    :param r: reward vector
    :param gamma: discount rate
    :param tol: tolerance for convergence
    :return: V: state-value vector
    """
    n_states, n_actions, _ = p.shape
    V_s = torch.zeros([n_states], device=p.device)        # The state value function

    err = 9.e9
    n_iters = 0

    if verbose:
        print("Starting value iteration ...")

    while err > tol and (max_iters is None or n_iters < max_iters):

        prev_V = V_s.clone()

        Q_sa = r[:, None] + gamma * torch.matmul(p, V_s)
        V_s = torch.max(Q_sa, dim=1)[0]

        err = torch.max(torch.abs(V_s-prev_V)).item()

        if verbose and n_iters % 10 == 0:
            print(f"Iteration {n_iters}\t err: {err}")
        n_iters += 1

    if verbose:
        print(f"Finished after {n_iters} iterations with err: {err}")

    return V_s, Q_sa


def value_iteration_sas_rewards(P_sas: torch.Tensor, r_sas: torch.Tensor, gamma=0.9, tol=0.01, max_iters=None,
                                q_init=None, verbose=False):
    """

    :param P_sas: [n_states] x [n_actions] x [n_states] tensor where p[s0, a, s1] is the probability of transitioning from
                s0 to s1 if action a is taken
    :param r_sas: [n_states] x [n_actions] x [n_states] tensor where r_sas[s0, a, s1] is the reward of transitioning from
                s0 to s1 if action a is taken
    :param gamma: discount rate
    :param tol: tolerance for convergence
    :return: V: state-value vector
    """

    n_states, n_actions, _ = P_sas.shape
    # The state-action value function
    Q_sa = torch.zeros(torch.Size([n_states, n_actions]), dtype=torch.float) if q_init is None else q_init

    err = 9.e9
    n_iters = 0

    if verbose:
        print("Starting value iteration ...")

    while err > tol and (max_iters is None or n_iters < max_iters):

        Q_new_sa = torch.sum(P_sas * (r_sas + gamma * torch.max(Q_sa, dim=1).values[None, None, :]), dim=2)

        err = torch.max(torch.abs(Q_new_sa - Q_sa).flatten())

        if verbose and n_iters % 10 == 0:
            print(f"Iteration {n_iters}\t err: {err}")
        n_iters += 1

        Q_sa = Q_new_sa

    if verbose:
        print(f"Finished after {n_iters} iterations with err: {err}")

    return Q_sa


def optimal_policy(Q: np.array):
    """
    :param Q: value function
    :return:
    """

    policy = np.argmax(Q, axis=1)

    return policy


def joint_transition(pi_sa: np.array, P_sas: np.array):
    """
    Computes the joint transition matrix for a given policy and transition matrix
    :param pi_sa: policy
    :param P_sas: transition matrix
    :return: joint transition matrix
    """
    pi_as = pi_sa.T
    P_ss = np.sum(P_sas * pi_as, axis=1)

    return P_ss


if __name__ == "__main__":

    # Example 1 D process
    r = np.array([0,0,0,0,10])
    p = np.array([
        [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
        [[1, 0, 0, 0, 0], [0, 0, 1, 0, 0]],
        [[0, 1, 0, 0, 0], [0, 0, 0, 1, 0]],
        [[0, 0, 1, 0, 0], [0, 0, 0, 0, 1]],
        [[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]],
    ])

    print("Reward: ")
    print(r)
    print("Transitions: ")
    print(p)
    V, Q = value_iteration(p, r)
    print(" ... done. Resulting state values:")
    print(V)
