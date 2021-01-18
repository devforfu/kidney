import os
import sys
import traceback
from dataclasses import dataclass
from typing import List, Callable, Dict

from sklearn.model_selection import KFold
from zeus.utils.misc import random_string, TimestampFormat

from kidney.datasets.kaggle import get_reader, SampleType
from run_utils import get_experiment_entry_point, override_argv_params_from_file


def main():
    entry_point, name = get_experiment_entry_point(with_name=True)
    override_argv_params_from_file()
    run_k_fold(entry_point, name, get_reader().get_keys(SampleType.Labeled))


def run_k_fold(entry_point: Callable, name: str, keys: List[str]):
    identifier = random_string(8)
    n_splits = int(os.environ.get("K_FOLD_SIZE", len(keys)))
    splits = KFold(n_splits).split(keys)
    for fold, (_, valid_idx) in enumerate(splits):
        fold_keys = ";".join([keys[i] for i in valid_idx])
        update_argv({
            "fold": fold_keys,
            "tags": f"id:{identifier},fold:{fold},key:{fold_keys},impl:{name}",
            "experiment_name": f"{name}_{identifier}",

            # requires explicit timestamp; otherwise, wandb continues
            # writing to the very first fold instead of creating a
            # new experiment
            "timestamp": TimestampFormat.VerboseShortSeconds.now(),

            "data_loader_multiprocessing_context": "spawn"
        })
        entry_point()


def update_argv(params: Dict[str, str]):
    sys.argv += [f"--{key}={value}" for key, value in params.items()]


@dataclass
class TracebackPrinter:
    separator_char: str = "-"
    separator_length: int = 80
    show_separators: bool = True

    def print_traceback(self, exception: Exception):
        self.print_separator()
        tb = traceback.format_tb(exception.__traceback__)
        for line in tb:
            print(line.strip())
        self.print_exception(exception)
        self.print_separator()

    @staticmethod
    def print_exception(exception: Exception):
        print(f"{exception.__class__.__name__}: {exception}")

    def print_separator(self):
        if self.show_separators:
            print(self.separator_char * self.separator_length)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard interruption request received. Terminating...")
    except Exception as e:
        TracebackPrinter().print_traceback(exception=e)
