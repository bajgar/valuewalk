from irl_algorithms.demonstrations import Demonstrations
from envs.gridworld import GridWorld


def plot_demo_counts(D: Demonstrations, env: GridWorld, title="State-action counts in demonstrations", onehot=True, **kwargs):
    counts = D.sa_counts(n_actions=env.n_actions, onehot=onehot, n_states=env.n_states)
    env.q_heatmap(q=counts, title=title, **kwargs)
