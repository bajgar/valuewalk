import pathlib
import argparse
import logging

import torch
import pyro.contrib.gp as gp
import matplotlib.pyplot as plt

from envs.gym_make import get_cartpole_env_config, get_lunar_lander_env_config
from experiments.birl.presets.priors import get_gp_prior, get_static_evals_gp_prior
from experiments.irl_experiment import IRLExperiment, IRLExperimentConfig
from experiments.paths import RESULTS_DIR, DATASET_DIR
from experiments.utils.file_names import get_result_file_path
from experiments.utils.load_avril_demos import get_load_avril_demos_factory
from irl_algorithms.avril import AVRIL
from irl_algorithms.demonstrations import Demonstrations
from irl_algorithms.demonstrations_config import BoltzmannDemosConfig
from irl_algorithms.mcmc_irl import BayesianIRLConfig
from irl_algorithms.value_walk import ValueWalkCts
from models.basic_models import mlp_factory

logging.basicConfig(level=logging.DEBUG)

def constant_mean_factory():
    # return lambda x: -torch.ones(x.shape[0])
    return lambda x: torch.zeros(x.shape[0])

def get_rbf_kernel_factory(input_dims):
    def rbf_kernel_factory():
        scale = 1.
        lengthscale = 0.03
        kernel = gp.kernels.RBF(input_dim=input_dims,
                                variance=torch.tensor(scale, requires_grad=False),
                                lengthscale=torch.tensor(lengthscale, requires_grad=False))
        return kernel
    return rbf_kernel_factory


# def aux_demo_factory(D, env, split=0):
#
#     D_aux = Demonstrations.load(DATASET_DIR / f"sbirl/volume/LunarLander-v2/aux_trajs_3t_split{split}.pkl")
#     D_aux.extend(D)
#
#     return D_aux

# def aux_demo_factory(D, env, split=0):
#
#     D_aux = Demonstrations()
#
#     for traj in D:
#
#
#
#
#     D_aux.extend(D)
#
#     return D_aux


def get_exp_config():

    # argparser = argparse.ArgumentParser()
    # argparser.add_argument('--num_trajs', type=int, default=1)
    #
    # args = argparser.parse_args()


    env_config = get_lunar_lander_env_config()

    demos_config = BoltzmannDemosConfig(
        env_name=env_config.env_name,
        demo_factory=get_load_avril_demos_factory,
        n_trajectories=1,
        beta_expert=3.,
        gamma=0.95,
        demo_subset_randomization=False,
        index_to_onehot=True
    )

    SMOKE_TEST = False

    irl_config = BayesianIRLConfig(
        irl_method_factory=ValueWalkCts,
        beta_expert=demos_config.beta_expert,
        gamma=demos_config.gamma,

        prior_mean_factory=constant_mean_factory,
        prior_kernel_factory=get_rbf_kernel_factory(input_dims=env_config.obs_dim + env_config.a_dim),
        reward_prior_factory=get_static_evals_gp_prior,

        final_q_to_r=False,

        warmup_steps=4000 if not SMOKE_TEST else 10,
        num_samples=10000 if not SMOKE_TEST else 10,
        checkpoint_frequency=None if not SMOKE_TEST else None,
        # num_chains=5,

        hmc_use_nuts=True,
        hmc_step_size=0.001,
        hmc_target_accept_prob=0.7,
        hmc_adapt_mass_matrix=False,
        pyro_jit_compile=True,

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

        # preprocessing_module_factory=None,
        q_model=mlp_factory(12, [24,24], 1),
        q_model_inputs=12,
        q_model_hidden_layer_sizes=[24,24],
        prior_scale=1.,
        # aux_demo_factory=aux_demo_factory
    )

    save_path = get_result_file_path(extra=f"{irl_config.num_samples}s_{demos_config.n_trajectories}t_oh_beta{int(irl_config.beta_expert)}_{'_'.join([str(s) for s in irl_config.q_model_hidden_layer_sizes]) if irl_config.q_model_hidden_layer_sizes else 'linear'}")

    exp_config = IRLExperimentConfig(
            env_config=env_config,
            irl_config=irl_config,
            demos_config=demos_config,
            result_save_path=save_path,
            save_method="torch")

    return exp_config


if __name__ == "__main__":

    # exp_config = get_exp_config()
    # experiment = IRLExperiment(exp_config)
    # reward_model, info = experiment.run()

    n_trajs = 7
    num_splits = 5
    num_reps_per_split = 2
    i = 0

    trajectory_nums = [15, 10]
    test_rewards = []

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--core_id", type=int, default=0)
    arg_parser.add_argument("--split", type=int, default=0)
    arg_parser.add_argument("--num_trajs", type=int, default=n_trajs)

    args = arg_parser.parse_args()

    split = args.split
    n_trajs = args.num_trajs

    # for n_trajs in trajectory_nums:
    #     for split in range(num_splits):
    #         for i in range(num_reps_per_split):
    print(f"Running {n_trajs} demos, split {split}, repetition {i} ...")
    exp_config = get_exp_config()
    exp_config.demos_config.n_trajectories = n_trajs
    exp_config.demos_config.data_split = "train"+str(split)
    # exp_config.irl_config.aux_demo_factory = lambda D, env: aux_demo_factory(D, env, split)
    exp_config.result_save_path = get_result_file_path(extra=f"gamma{exp_config.irl_config.gamma:.2}_beta{exp_config.irl_config.beta_expert:.1}_{n_trajs}t_split{split}_rep{i}_hl{'_'.join([str(size) for size in exp_config.irl_config.q_model_hidden_layer_sizes])}_scale1_lsp03_c{args.core_id}_paper")
    experiment = IRLExperiment(exp_config)
    reward_model, info = experiment.run()
