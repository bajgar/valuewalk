import glob
from pathlib import Path

import torch

from analysis.result_loading import load_and_combine_vw_models
from experiments.paths import RESULTS_DIR
from experiments.utils.evaluation import test_vw_apprentice_on_gym_env


ENV_NAME = "LunarLander-v2"
RESULTS_GLOB = "birl/06b_vw_lunar_lander*_hypersearch_*.pt"
DEFAULT_HIDDENS = [16]  # Hiddens will be parsed from file name if available
INPUT_DIM = 12

EVALUATE_CHECKPOINTS = False


def evaluate_models():
    """
    Evaluate all the VW models in the results directory on the ENV_NAME environment and save results.
    """
    all_results = [file_name for file_name in glob.glob(str(RESULTS_DIR / RESULTS_GLOB)) if
                   "info" not in file_name and 'evaluation' not in file_name]

    if not EVALUATE_CHECKPOINTS:
        all_results = [result for result in all_results if "checkpoint" not in result]
    else:
        all_results = [result for result in all_results if "checkpoint" in result]

    for result_file in all_results:

        # If the evaluation file already exists, skip it
        if Path(result_file.replace(".pt", ".evaluation.pt")).exists():
            print(f"Evaluation file already exists for {result_file}. Skipping.")
            continue

        print(f"Evaluating {result_file}")
        model = load_and_combine_vw_models([result_file], inputs=INPUT_DIM, default_hiddens=DEFAULT_HIDDENS, outputs=1)

        evaluations = {
            "mean": test_vw_apprentice_on_gym_env(model, env_name=ENV_NAME,
                                                  aggregation_fn=lambda x: torch.mean(x, dim=-1)),
            "median": test_vw_apprentice_on_gym_env(model, env_name=ENV_NAME,
                                                    aggregation_fn=lambda x: torch.median(x, dim=-1)[0]),
            ".2q": test_vw_apprentice_on_gym_env(model, env_name=ENV_NAME,
                                                 aggregation_fn=lambda x: torch.quantile(x, 0.2, dim=-1)),
        }

        save_file = result_file.replace(".pt", ".evaluation.pt")
        torch.save(evaluations, save_file)
        print(f"Saved evaluation to {save_file}")


if __name__ == "__main__":
    evaluate_models()
