import json
from argparse import ArgumentParser

from kidney.cli import extend_parser, comma_separated_list_of_strings


@extend_parser
def add_training_loop_args(parser: ArgumentParser) -> ArgumentParser:
    """Extends arguments parser with training parameters."""

    parser.add_argument(
        "-e", "--epochs",
        default=1,
        type=int,
        help="Number of training epochs."
    )
    parser.add_argument(
        "-bs", "--batch_size",
        default=4,
        type=int,
        help="Data loader batch size."
    )
    parser.add_argument(
        "-lr", "--learning_rate",
        default=1e-4,
        type=float,
        help="Learning rate."
    )
    parser.add_argument(
        "-wd", "--weight_decay",
        default=0.0,
        type=float,
        help=
        "Weight decay; the option is ignored in case if a given "
        "optimizer doesn't support weigh decay or ignores this "
        "option."
    )
    parser.add_argument(
        "--fold_no",
        default=0,
        type=int,
        help=
        "Data fold to use for training; the interpretation of this argument is up "
        "to training code."
    )
    return parser


@extend_parser
def add_optimizer_args(parser: ArgumentParser) -> ArgumentParser:
    """Extends arguments parser with optimizer-specific parameters."""

    parser.add_argument(
        "--optimizer_name",
        default="adam",
        help="Optimizer name."
    )
    parser.add_argument(
        "--optimizer_config",
        default="{}",
        type=json.loads,
        help="Optimizer-specific parameters as JSON string."
    )
    return parser


@extend_parser
def add_scheduler_args(parser: ArgumentParser) -> ArgumentParser:
    """Extends arguments parser with scheduler-specific parameters."""

    parser.add_argument(
        "--scheduler_name",
        default=None,
        help="Scheduler name"
    )
    parser.add_argument(
        "--scheduler_interval",
        default="epoch",
        choices=["epoch", "step"],
        help="How often to call scheduler: per epoch or per step (training batch)."
    )
    parser.add_argument(
        "--scheduler_config",
        default="{}",
        type=json.loads,
        help="Scheduler-specific parameters as JSON string."
    )
    return parser


@extend_parser
def add_loss_args(parser: ArgumentParser) -> ArgumentParser:
    """Extends arguments parser with loss function parameters."""

    parser.add_argument(
        "--loss_name",
        required=True,
        help="Loss function."
    )
    parser.add_argument(
        "--loss_config",
        default="{}",
        type=json.loads,
        help="Loss function parameters as JSON string."
    )
    return parser


@extend_parser
def add_metrics_args(parser: ArgumentParser) -> ArgumentParser:
    """Extends arguments parser with metric functions parameters."""

    parser.add_argument(
        "--metrics",
        type=comma_separated_list_of_strings,
        default=None,
        help="A list of comma-separated metric names."
    )
    return parser


@extend_parser
def add_validation_args(parser: ArgumentParser) -> ArgumentParser:
    """Extends arguments parser with validation-specific parameters."""

    parser.add_argument(
        "--fold",
        default=None,
        help=
        "A parameter that defines validation approach; it's "
        "implementation is up to specific training script."
    )
    return parser
