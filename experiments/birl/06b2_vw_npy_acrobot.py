import argparse

import torch
from envs.gym_make import get_cartpole_env_config, get_lunar_lander_env_config, get_acrobot_env_config
from experiments.birl.presets.priors import get_static_evals_gp_prior_jax
from experiments.irl_experiment import IRLExperiment, IRLExperimentConfig
from experiments.utils.file_names import get_result_file_path
from experiments.utils.load_avril_demos import get_load_avril_demos_factory
from irl_algorithms.avril import AVRIL
from irl_algorithms.demonstrations_config import BoltzmannDemosConfig
from irl_algorithms.mcmc_irl import BayesianIRLConfig
from irl_algorithms.value_walk import VWCN, VWCNBJ
from models.basic_models_jax import mlp_factory_jax
from gpjax.mean_functions import Zero
from gpjax.kernels import RBF
import jax.numpy as jnp


def zero_mean_factory():
    return Zero()


def get_rbf_kernel_factory():
    scale = 1.
    lengthscale = 0.5
    return RBF(variance=jnp.array(scale),
                lengthscale=jnp.array(lengthscale))


def get_exp_config():

    # argparser = argparse.ArgumentParser()
    # argparser.add_argument('--num_trajs', type=int, default=1)
    #
    # args = argparser.parse_args()

    env_config = get_acrobot_env_config()

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
    BLACKJAX = True

    irl_config = BayesianIRLConfig(
        irl_method_factory=VWCN if not BLACKJAX else VWCNBJ,
        beta_expert=demos_config.beta_expert,
        gamma=demos_config.gamma,

        prior_mean_factory=zero_mean_factory,
        prior_kernel_factory=get_rbf_kernel_factory,
        reward_prior_factory=get_static_evals_gp_prior_jax,

        final_q_to_r=False,

        warmup_steps=2000 if not SMOKE_TEST else 10,
        num_samples=5000 if not SMOKE_TEST else 10,

        hmc_use_nuts=True,
        hmc_step_size=0.01,
        hmc_target_accept_prob=0.7,
        hmc_adapt_mass_matrix=False,
        pyro_jit_compile=True,

        # preprocessing_module_factory=None,
        q_model=mlp_factory_jax(9, [16], 1),
        q_model_inputs=9,
        q_model_hidden_layer_sizes=[16],
        prior_scale=1.,
    )

    exp_config = IRLExperimentConfig(
            env_config=env_config,
            irl_config=irl_config,
            demos_config=demos_config,
            result_save_path=get_result_file_path(extra=f"{irl_config.num_samples}s_{demos_config.n_trajectories}t_oh_fixed"),
            save_method="torch")

    return exp_config


if __name__ == "__main__":
    #
    # exp_config = get_exp_config()
    # experiment = IRLExperiment(exp_config)
    # reward_model, info = experiment.run()

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--split", type=int, default=0)
    args = arg_parser.parse_args()
    split = args.split

    num_repetitions = 4

    trajectory_nums = [1, 3, 7, 10, 15]
    test_rewards = []

    for n in trajectory_nums:
        test_rewards_n = []
        for i in range(num_repetitions):
            print(f"Running {n} demos, repetition {i}...")
            exp_config = get_exp_config()
            exp_config.demos_config.n_trajectories = n
            exp_config.demos_config.data_split = "train" + str(split)

            exp_config.result_save_path = get_result_file_path(extra=f"{n}t_hl{'_'.join([str(size) for size in exp_config.irl_config.q_model_hidden_layer_sizes])}_split{split}_paper")
            experiment = IRLExperiment(exp_config)
            reward_model, info = experiment.run()
