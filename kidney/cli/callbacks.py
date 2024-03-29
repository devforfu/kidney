from argparse import ArgumentParser

from kidney.cli import extend_parser

DEFAULT_TRACKED_METRIC = "val_loss"
DEFAULT_METRIC_MODE = "min"


@extend_parser
def add_early_stopping_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--early_stopping_enabled",
        action="store_true",
        help="Enable early stopping callback."
    )
    parser.add_argument(
        "--early_stopping_metric",
        default=DEFAULT_TRACKED_METRIC,
        help="Metric to track when deciding for early stopping."
    )
    parser.add_argument(
        "--early_stopping_mode",
        default=DEFAULT_METRIC_MODE,
        choices=["min", "max"],
        help=
        "If mode=min, then check if metric decreases; "
        "check if metric increases otherwise."
    )
    parser.add_argument(
        "--early_stopping_patience",
        default=1,
        type=int,
        help="Number of epochs to wait for metrics improvement until stopping."
    )
    parser.add_argument(
        "--early_stopping_best_weights",
        default=False,
        type=bool,
        help="Restore best model weights if metric didn't improve."
    )
    return parser


@extend_parser
def add_checkpointing_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--checkpoints_enabled",
        action="store_true",
        help="Enable checkpoints callback."
    )
    parser.add_argument(
        "--checkpoints_metric",
        default=DEFAULT_TRACKED_METRIC,
        help="Metric to track when deciding to make a checkpoint."
    )
    parser.add_argument(
        "--checkpoints_mode",
        default=DEFAULT_METRIC_MODE,
        choices=["min", "max"]
    )
    parser.add_argument(
        "--checkpoints_top_k",
        type=int,
        default=1,
        help="Number of best checkpoints to save."
    )
    parser.add_argument(
        "--checkpoints_path",
        type=str,
        default=None,
        help=
        "Filepath where a checkpoint is saved (can include formatting to "
        "save each epoch into a new file); if not provided, should be "
        "derived from other experiment's options."
    )
    return parser
