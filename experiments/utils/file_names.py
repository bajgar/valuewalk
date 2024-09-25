import inspect
import logging
import pathlib
import datetime

from experiments.paths import EXPERIMENTS_ROOT, RESULTS_DIR, SOURCE_ROOT


def get_result_file_path(extra: str = None, suffix='.pt') -> pathlib.Path:
    """Returns the path to the file where the results will be saved. It's the file with the same relative path to
    RESULTS_DIR as the file where this function is called has to the 'experiments' dir. With the following changes:
    - The extension is replaced by suffix (default: '.pt').
    - If extra is provided, it is appended to the file name.
    - The file name is appended with the current date and time in the format _YYMMDDHHMMSS.

    Returns:
        Path: File where the results will be saved.
    """
    caller_file_name = pathlib.Path(inspect.stack()[1].filename)

    try:
        caller_file_name = caller_file_name.relative_to(EXPERIMENTS_ROOT)
    except ValueError:
        logging.warning(f"File {caller_file_name} is not under {EXPERIMENTS_ROOT}. Saving relative to SOURCE_ROOT.")
        try:
            caller_file_name = caller_file_name.relative_to(SOURCE_ROOT)
        except ValueError:
            logging.warning(f"File {caller_file_name} is not under {SOURCE_ROOT}. Saving directly to RESULTS_DIR.")
            caller_file_name = caller_file_name.name

    # Position this relative to RESULTS_DIR.
    results_file = RESULTS_DIR / caller_file_name

    # Strip the extension.
    results_file = results_file.with_suffix(suffix)

    # Append extra to the file name.
    if extra is not None:
        results_file = results_file.with_stem(f"{results_file.stem}_{extra}")

    # Append date and time to the file name in the format _YYMMDDHHMMSS.
    results_file = results_file.with_stem(f"{results_file.stem}_{datetime.datetime.now():%y%m%d-%H%M%S}")

    # Create the parent directories if they don't exist.
    results_file.parent.mkdir(parents=True, exist_ok=True)

    return results_file
