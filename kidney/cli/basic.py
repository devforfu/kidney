from argparse import ArgumentParser, Namespace
from multiprocessing import cpu_count
from pathlib import Path

from zeus.utils import home, TimestampFormat


def basic_parser(script_filename: str = None) -> ArgumentParser:
    """Creates basic CLI parser with generic parameters to configure training process.

    The parser is expected to be used on its own or as a foundation for more complex
    sets of arguments. Though it could be used in any setup, it was tailored to accompany
    PyTorch Lightning experiments. It includes the most generic meta-information or
    technical specifications for the experiment.

    Args:
        script_filename
            Path to the training script. If provided, used to
            automatically derive some default values for parameters.

    Returns:
        parser: The constructed argument parser.

    """
    experiment_name = (
        Path(script_filename).stem
        if script_filename is not None
        else None
    )

    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        help=
        "A string, filesystem path, or any other identifier defining "
        "the dataset; it's exact usage is up to the training code."
    )
    parser.add_argument(
        "--tags",
        type=lambda s: s.split(','),
        default=None,
        help=
        "Optional comma-separated string of tags storing additional "
        "meta-information about the training code."
    )
    parser.add_argument(
        "--experiment_name",
        default=experiment_name,
        help=
        "Experiment name; used for logging and checkpointing purposes"
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help=
        "A name of a specific experiment's execution; helpful for logging "
        "purposes to distinguish runs of the same experiment but with the "
        "different parameters."
    )
    parser.add_argument(
        "--experiment_dir",
        default=home("experiments"),
        help=
        "A folder to store experiment's artifacts and meta-data."
    )
    parser.add_argument(
        "--num_workers",
        default=cpu_count(),
        help="A number of workers for multi-processing parts of the training "
             "pipeline (i.e., data loaders)."
    )
    parser.add_argument(
        "--timestamp",
        default=TimestampFormat.VerboseShortSeconds.now(),
        help="Experiment's run timestamp."
    )
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        help="Random generator seed."
    )
