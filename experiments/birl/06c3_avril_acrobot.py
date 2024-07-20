import argparse
import pathlib

import torch
import pyro.contrib.gp as gp
import matplotlib.pyplot as plt

from envs.gym_make import get_cartpole_env_config, get_lunar_lander_env_config, get_acrobot_env_config
from experiments.birl.presets.priors import get_gp_prior
from experiments.irl_experiment import IRLExperiment, IRLExperimentConfig
from experiments.paths import RESULTS_DIR
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

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num_trajs', type=int, default=1)

    args = argparser.parse_args()

    env_config = get_acrobot_env_config()

    demos_config = BoltzmannDemosConfig(
        env_name=env_config.env_name,
        demo_factory=get_load_avril_demos_factory,
        n_trajectories=args.num_trajs,
        beta_expert=1.,
        gamma=0.9,
        demo_subset_randomization=False,
        index_to_onehot=True
    )

    SMOKE_TEST = False

    irl_config = BayesianIRLConfig(
        irl_method_factory=AVRIL,
        beta_expert=1.,
        gamma=0.99,
        prior_mean_factory=zero_mean_factory,
        prior_kernel_factory=get_rbf_kernel_factory(input_dims=env_config.obs_dim + env_config.a_dim),
        reward_prior_factory=get_gp_prior,

        final_q_to_r=True,

        svi_iters=10,
        svi_lr=1e-4,
        svi_reporting_frequency=500,

        num_synth_trajectories=0,
        max_synth_traj_length=25,
        num_action_samples=20,

        # preprocessing_module_factory=None,
        q_model=mlp_factory(9, [16], 1),
        q_model_inputs=9,
        q_model_hidden_layer_sizes=[16],
        prior_scale=1.,

        state_only=True
    )

    exp_config = IRLExperimentConfig(
            env_config=env_config,
            irl_config=irl_config,
            demos_config=demos_config,
            result_save_path=get_result_file_path(extra=f"{irl_config.num_samples}s_{demos_config.n_trajectories}t_oh_fixed"),
            save_method="torch")

    return exp_config


if __name__ == "__main__":

    exp_config = get_exp_config()
    experiment = IRLExperiment(exp_config)
    reward_model, info = experiment.run()
