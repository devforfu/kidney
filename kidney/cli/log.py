from argparse import ArgumentParser

from kidney.cli import extend_parser


@extend_parser
def add_logging_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--logging_steps",
        default=1,
        type=int,
        help="Logging frequency."
    )
    parser.add_argument(
        '--wandb_logging_enabled',
        action='store_true',
        help="Enable W&B logging."
    )
    return parser
