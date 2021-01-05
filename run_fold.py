import sys
from datetime import datetime
from typing import List, Callable, Dict

from zeus.utils.misc import random_string, TimestampFormat

from kidney.datasets.kaggle import get_reader, SampleType
from run_utils import get_experiment_entry_point, override_argv_params_from_file


def main():
    entry_point, name = get_experiment_entry_point(with_name=True)
    override_argv_params_from_file()
    run_k_fold(entry_point, name, get_reader().get_keys(SampleType.Labeled))


def run_k_fold(entry_point: Callable, name: str, keys: List[str]):
    identifier = random_string(8)
    for fold, key in enumerate(keys, 1):
        update_argv({
            "fold": key,
            "tags": f"id:{identifier},fold:{fold},key:{key},impl:{name}",
            "experiment_name": identifier,

            # requires explicit timestamp; otherwise, wandb continues
            # writing to the very first fold instead of creating a
            # new experiment
            "timestamp": TimestampFormat.VerboseShortSeconds.now()
        })
        entry_point()


def update_argv(params: Dict[str, str]):
    sys.argv += [f"--{key}={value}" for key, value in params.items()]


if __name__ == '__main__':
    main()
