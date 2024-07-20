import pathlib

import torch
import pyro.contrib.gp as gp
import matplotlib.pyplot as plt

from envs.gym_make import get_cartpole_env_config
from experiments.birl.presets.priors import get_gp_prior
from experiments.irl_experiment import IRLExperiment, IRLExperimentConfig
from experiments.paths import RESULTS_DIR
from experiments.utils.file_names import get_result_file_path
from experiments.utils.load_avril_demos import get_load_avril_demos_factory
from irl_algorithms.avril import AVRIL
from irl_algorithms.demonstrations_config import BoltzmannDemosConfig
from irl_algorithms.mcmc_irl import BayesianIRLConfig


"""
This runs AVRIL for varying numbers of demonstration trajectories to reproduce Figure 3 from the paper.
"""


def zero_mean_factory():
    return lambda x: torch.zeros(x.shape[0])


def get_rbf_kernel_factory(input_dims):
    def rbf_kernel_factory():
        scale = 1.
        lengthscale = 0.2
        kernel = gp.kernels.RBF(input_dim=input_dims,
                                variance=torch.tensor(scale),
                                lengthscale=torch.tensor(lengthscale))
        return kernel
    return rbf_kernel_factory


def get_exp_config():
    env_config = get_cartpole_env_config()

    demos_config = BoltzmannDemosConfig(
        env_name=env_config.env_name,
        demo_factory=get_load_avril_demos_factory,
        n_trajectories=15,
        beta_expert=1.,
        gamma=0.99,
        demo_subset_randomization=True
    )

    irl_config = BayesianIRLConfig(
        irl_method_factory=AVRIL,
        beta_expert=1.,
        prior_mean_factory=zero_mean_factory,
        prior_kernel_factory=get_rbf_kernel_factory(input_dims=env_config.obs_dim + env_config.a_dim),
        reward_prior_factory=get_gp_prior,

        svi_iters=20000,
        svi_lr=1e-3,
        svi_reporting_frequency=500,

        num_action_samples=10,

        # preprocessing_module_factory=None,
        q_model_inputs=4,
        q_model_hidden_layer_sizes=[64, 64],
        prior_scale=1.,

        state_only=True
    )

    exp_config = IRLExperimentConfig(
            env_config=env_config,
            irl_config=irl_config,
            demos_config=demos_config,
            result_save_path=get_result_file_path(extra=f"{demos_config.n_trajectories}t_hl{'_'.join([str(size) for size in irl_config.q_model_hidden_layer_sizes])}"),
            save_method="torch")

    return exp_config


if __name__ == "__main__":

    num_repetitions = 10

    trajectory_nums = [1, 3, 7, 10, 15]
    # trajectory_nums = [10]
    test_rewards = []

    for n in trajectory_nums:
        test_rewards_n = []
        for i in range(num_repetitions):
            exp_config = get_exp_config()
            exp_config.demos_config.n_trajectories = n
            exp_config.result_save_path = get_result_file_path(extra=f"{n}t_hl{'_'.join([str(size) for size in exp_config.irl_config.q_model_hidden_layer_sizes])}")
            experiment = IRLExperiment(exp_config)
            reward_model, info = experiment.run()
            test_rewards_n.append(info["test_mean_reward"])
            print(f"{n} demos, \t test reward: {info['test_mean_reward']}")
        test_rewards.append(sum(test_rewards_n) / num_repetitions)
        print(f"Average test reward for {n} trajs: \t {test_rewards[-1]}")

    plt.plot(trajectory_nums, test_rewards)
    plt.xlabel("Number of Demonstrations")
    plt.ylabel("Test Reward")
    plt.savefig("avril_cartpole.png")
    plt.show()
