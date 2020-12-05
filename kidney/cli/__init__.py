from argparse import ArgumentParser
from functools import reduce, wraps
from typing import Callable, Optional, List

import pytorch_lightning as pl


def create_parser(
    script_filename: str,
    basic_parser: Callable,
    extensions: Optional[List[Callable]] = None,
) -> ArgumentParser:
    if extensions is None:
        parser = basic_parser(script_filename)
    else:
        parser = reduce(lambda extend, previous: extend(previous), extensions)
    return parser


def default_parser(script_filename: str) -> ArgumentParser:
    """Creates default parser using all extensions defined in the project.

    The default parser also includes all the options from PyTorch Lightning trainer.

    Returns:
        parser: The CLI parser that includes all available options.

    """
    from kidney.cli.basic import basic_parser
    from kidney.cli import training
    return create_parser(script_filename, basic_parser, extensions=[
        training.add_training_loop_args,
        training.add_optimizer_args,
        training.add_scheduler_args,
        pl.Trainer.add_argparse_args,
    ])


def extend_parser(func: Callable):
    @wraps(func)
    def wrapper(parser: ArgumentParser) -> ArgumentParser:
        return func(ArgumentParser(parents=[parser], add_help=False))
    return wrapper
