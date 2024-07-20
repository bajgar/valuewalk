import pathlib
import os

_results_root = os.environ.get("RESULTS_ROOT", os.path.expanduser("~/results"))
RESULTS_DIR = pathlib.Path(_results_root) / "irl-torch"

_dataset_root = os.environ.get("DATASET_ROOT", os.path.expanduser("~/datasets"))
DATASET_DIR = pathlib.Path(_dataset_root) / "irl-torch"

SOURCE_ROOT = pathlib.Path(__file__).parent.parent
EXPERIMENTS_ROOT = SOURCE_ROOT / "experiments"
