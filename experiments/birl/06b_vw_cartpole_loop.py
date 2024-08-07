import pathlib
import argparse

import torch
import pyro.contrib.gp as gp
import matplotlib.pyplot as plt

from envs.gym_make import get_cartpole_env_config
from experiments.birl.presets.priors import get_gp_prior, get_static_evals_gp_prior
from experiments.irl_experiment import IRLExperiment, IRLExperimentConfig
from experiments.paths import RESULTS_DIR
from experiments.utils.evaluation import test_vw_apprentice_on_gym_env
from experiments.utils.file_names import get_result_file_path
from experiments.utils.load_avril_demos import get_load_avril_demos_factory
from irl_algorithms.avril import AVRIL
from irl_algorithms.demonstrations_config import BoltzmannDemosConfig
from irl_algorithms.mcmc_irl import BayesianIRLConfig
from irl_algorithms.value_walk import ValueWalkCts
from models.basic_models import mlp_factory


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
        n_trajectories=1,
        beta_expert=3.,
        gamma=0.95,
        # demo_subset_randomization=True,
        index_to_onehot=True
    )

    irl_config = BayesianIRLConfig(
        irl_method_factory=ValueWalkCts,
        beta_expert=demos_config.beta_expert,
        gamma=demos_config.gamma,
        prior_mean_factory=zero_mean_factory,
        prior_kernel_factory=get_rbf_kernel_factory(input_dims=env_config.obs_dim + env_config.a_dim),
        reward_prior_factory=get_static_evals_gp_prior,

        num_samples=2000,
        warmup_steps=1000,

        hmc_use_nuts=True,
        hmc_step_size=0.01,
        hmc_target_accept_prob=0.7,
        hmc_adapt_mass_matrix=False,
        pyro_jit_compile=True,

        # preprocessing_module_factory=None,
        q_model=mlp_factory(6, [8], 1),
        q_model_inputs=6,
        q_model_hidden_layer_sizes=[8],
        prior_scale=1.,

        state_only=True
    )

    exp_config = IRLExperimentConfig(
            env_config=env_config,
            irl_config=irl_config,
            demos_config=demos_config,
            result_save_path=get_result_file_path(extra=f"{irl_config.num_samples}s_{demos_config.n_trajectories}t_oh"),
            save_method="torch")

    return exp_config


if __name__ == "__main__":

    # exp_config = get_exp_config()
    # experiment = IRLExperiment(exp_config)
    # reward_model, info = experiment.run()

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--split", type=int, default=0)
    args = arg_parser.parse_args()
    split = args.split

    num_repetitions = 5

    trajectory_nums = [15]
    test_rewards = []
    i=0

    for n in trajectory_nums:
        test_rewards_n = []
        for split in range(num_repetitions):
            print(f"Running {n} demos, repetition {i}...")
            exp_config = get_exp_config()
            exp_config.demos_config.n_trajectories = n
            exp_config.demos_config.data_split = "train" + str(split)

            exp_config.result_save_path = get_result_file_path(extra=f"{n}t_hl{'_'.join([str(size) for size in exp_config.irl_config.q_model_hidden_layer_sizes])}_split{split}_paper")
            experiment = IRLExperiment(exp_config)
            reward_model, info = experiment.run()

            env_name = exp_config.env_config.env_name
            reward_model.q_model = mlp_factory(6, [8], 1)
            evaluations = {
                "mean": test_vw_apprentice_on_gym_env(reward_model, env_name=env_name,
                                                      aggregation_fn=lambda x: torch.mean(x, dim=-1)),
                "median": test_vw_apprentice_on_gym_env(reward_model, env_name=env_name,
                                                        aggregation_fn=lambda x: torch.median(x, dim=-1)[0]),
                ".2q": test_vw_apprentice_on_gym_env(reward_model, env_name=env_name,
                                                     aggregation_fn=lambda x: torch.quantile(x, 0.2, dim=-1)),
            }

            # test_rewards_n.append(info["test_mean_reward"])
            # print(f"{n} demos, \t test reward: {info['test_mean_reward']}")
        # test_rewards.append(sum(test_rewards_n) / num_repetitions)
        # print(f"Average test reward for {n} trajs: \t {test_rewards[-1]}")
    #
    # plt.plot(trajectory_nums, test_rewards)
    # plt.xlabel("Number of Demonstrations")
    # plt.ylabel("Test Reward")
    # plt.savefig("avril_cartpole.png")
    # plt.show()
