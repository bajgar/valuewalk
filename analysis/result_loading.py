import glob
from pathlib import Path
from typing import Any, List
import re

import torch

from experiments.paths import RESULTS_DIR
from models.basic_models import mlp_factory
from models.reward_models.q_based_reward_model import QBasedSampleBasedRewardModel


def get_result_file_paths(experiment_partial_path: str, suffix=".pt") -> List[str]:
    results_glob = RESULTS_DIR / (experiment_partial_path + "*" + suffix)
    results_files = glob.glob(str(results_glob))
    results_files.sort()
    return results_files


def load_result_files(path_list: List[str]) -> List[Any]:
    results = [torch.load(path) for path in path_list]
    return results


def parse_hidden_layers_from_filename(filename, default_hiddens=(16,)):
    # Pattern to match a variable number of hidden layer sizes (e.g., "64_32_16") in the filename
    # This pattern looks for the sequence after "beta" and before the date-time stamp, allowing for any number of underscore-separated digits.
    pattern = r'_hl((?:\d+_)*\d+)_(?:scale\d_)?(?:c\d_)?(?:\w+_)?\d{6}-\d{6}\.'
    match = re.search(pattern, str(filename))
    if match:
        # Extract the matched group containing the hidden layer sizes
        hidden_layers_str = match.group(1)
        # Split the sizes into a list of integers
        hidden_layers = [int(size) for size in hidden_layers_str.split('_')]
        return hidden_layers
    else:
        print(f"No matching hidden layer sizes found in the filename. Using the default of {default_hiddens}.")
        return default_hiddens


def load_vw_model(model_file: str | Path) -> QBasedSampleBasedRewardModel:

    if "checkpoint" in model_file:
        samples = torch.load(model_file, map_location=torch.device('cpu'))['samples']
        model = QBasedSampleBasedRewardModel(q_param_samples=samples,
                                             q_model=None)
        if len(samples['theta_q'].shape) == 1:
            model.q_param_samples['theta_q'] = model.q_param_samples['theta_q'].unsqueeze(0)
    else:
        model = torch.load(model_file, map_location=torch.device('cpu'))

    return model


def load_and_combine_vw_models(model_files: List[str | Path], subsample : int = None, max_samples: int = 2000, inputs=12,
                               default_hiddens=(16,), outputs=1) -> QBasedSampleBasedRewardModel:
    """
    Load saved VW models from result files and combine the samples into a single RewardModel.
    Only models with the same parameters (e.g., hidden layer sizes) can be combined.

    Args:
        model_file:
        subsample: Subsamples the MCMC samples to reduce the number of samples. If None, no subsampling is done.
        max_samples: Maximum number of samples to keep. If the number of samples exceeds this value, the samples are
                    subsampled. (ignored if subsample is not None)
        default_hiddens:

    Returns:

    """

    hiddens = parse_hidden_layers_from_filename(model_files[0], default_hiddens=default_hiddens)

    base_model = load_vw_model(model_files[0])

    for model_file in model_files[1:]:
        assert hiddens == parse_hidden_layers_from_filename(model_file, default_hiddens=default_hiddens), \
            "Hidden layer sizes must match for all models to be combined."
        model = torch.load(model_file, map_location=torch.device('cpu'))
        base_model.q_param_samples['theta_q'] = torch.cat([base_model.q_param_samples['theta_q'], model.q_param_samples['theta_q']], dim=0)

    base_model.q_model = mlp_factory(inputs, hiddens, outputs)  # Adjust these parameters as needed

    if subsample is None and base_model.q_param_samples['theta_q'].shape[0] > max_samples:
        subsample = base_model.q_param_samples['theta_q'].shape[0] // max_samples

    if subsample:
        base_model.q_param_samples['theta_q'] = base_model.q_param_samples['theta_q'][::subsample]

    # print(f"Loaded model with param shape {base_model.q_param_samples['theta_q'].shape}")

    return base_model
