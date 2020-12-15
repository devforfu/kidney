import importlib
import sys


def main():
    entry_point = get_experiment_entry_point()
    override_argv_params_from_file()
    entry_point()


def get_experiment_entry_point():
    experiment_name = sys.argv[1]
    module = importlib.import_module(f"kidney.experiments.{experiment_name}")
    return getattr(module, "main")


def override_argv_params_from_file():
    filename = sys.argv[2]
    with open(filename) as fp:
        lines = [line.strip() for line in fp]
    sys.argv = [sys.argv[0]] + lines


if __name__ == '__main__':
    main()
