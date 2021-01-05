import sys
from typing import List, Callable, Dict

from zeus.utils.misc import random_string

from kidney.datasets.kaggle import get_reader, SampleType
from run_utils import get_experiment_entry_point, override_argv_params_from_file


def main():
    entry_point = get_experiment_entry_point()
    override_argv_params_from_file()
    run_k_fold(entry_point, get_reader().get_keys(SampleType.Labeled))


def run_k_fold(entry_point: Callable, keys: List[str]):
    identifier = random_string(8)
    for fold, key in enumerate(keys, 1):
        update_argv({
            "fold": key,
            "tags": f"id:{identifier},fold:{fold},key:{key}"
        })
        entry_point()


def update_argv(params: Dict[str, str]):
    sys.argv += [f"--{key}={value}" for key, value in params.items()]


if __name__ == '__main__':
    main()
