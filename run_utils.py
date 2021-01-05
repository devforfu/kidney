import importlib
import sys


def get_experiment_entry_point(with_name: bool = False):
    experiment_name = sys.argv[1]
    module = importlib.import_module(f"kidney.experiments.{experiment_name}")
    entry_point = getattr(module, "main")
    return entry_point, experiment_name if with_name else entry_point


def override_argv_params_from_file():
    filename = sys.argv[2]
    with open(filename) as fp:
        lines = [line.strip() for line in fp]
    sys.argv = [sys.argv[0]] + lines
