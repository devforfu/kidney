from argparse import ArgumentParser
from functools import reduce, wraps
from typing import Callable, Optional, List

import pytorch_lightning as pl

from kidney.experiments import save_experiment_info
from kidney.log import get_logger
from kidney.parameters import as_attribute_dict


def create_parser(
    script_filename: str,
    basic_parser: Callable,
    extensions: Optional[List[Callable]] = None,
) -> ArgumentParser:
    parser = basic_parser(script_filename)
    if extensions is not None:
        parser = reduce(lambda previous, extend: extend(previous), extensions, parser)
    return parser


def default_args() -> List[Callable]:
    from kidney.cli import training, callbacks, log
    return [
        training.add_training_loop_args,
        training.add_optimizer_args,
        training.add_scheduler_args,
        training.add_loss_args,
        training.add_metrics_args,
        callbacks.add_early_stopping_args,
        callbacks.add_checkpointing_args,
        log.add_logging_args,
        pl.Trainer.add_argparse_args,
    ]


def create_default_parser(script_filename: str) -> ArgumentParser:
    """Creates default parser using all extensions defined in the project.

    The default parser also includes all the options from PyTorch Lightning trainer.

    Returns:
        parser: The CLI parser that includes all available options.

    """
    from kidney.cli.basic import basic_parser
    return create_parser(script_filename, basic_parser, extensions=default_args())


def extend_parser(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(parser: ArgumentParser) -> ArgumentParser:
        return func(ArgumentParser(parents=[parser], add_help=False))
    return wrapper


def entry_point(base_parser_factory: Callable, extensions: Optional[List[Callable]] = None):

    def wrapper(func: Callable):
        parser = base_parser_factory()

        if extensions is not None:
            parser = reduce(lambda previous, extend: extend(previous), extensions, parser)

        @wraps(func)
        def wrapped():
            result = func(as_attribute_dict(parser.parse_args()))
            if result is not None:
                trainer, info = result
                logger = get_logger(__file__)
                logger.info("saving experiment parameters into checkpoints folder")
                params_dir = save_experiment_info(trainer, info)
                if params_dir is None:
                    logger.warning("checkpoints dir is not found; will not save info")
                else:
                    logger.info("experiment artifacts saved into the folder: %s", params_dir)

        return wrapped

    return wrapper


def comma_separated_list_of_strings(value: str) -> List[str]:
    return value.split(',')


def comma_separated_list_of_integers(value: str) -> List[int]:
    return list(map(int, comma_separated_list_of_strings(value)))

