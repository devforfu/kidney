from argparse import ArgumentParser

import json


def add_training_args(parser: ArgumentParser) -> ArgumentParser:
    """Extends arguments parser with training parameters."""

    parser = ArgumentParser(parents=[parser], add_help=False)
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Data loader batch size."
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="Learning rate."
    )
    parser.add_argument(
        "--logging_steps",
        default=1,
        type=int,
        help="Logging frequency."
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


def add_optimizer_args(parser: ArgumentParser) -> ArgumentParser:
    """Extends arguments parser with optimizer-specific parameters."""

    parser = ArgumentParser(parents=[parser], add_help=False)
    parser.add_argument(
        "--optimizer_name",
        default="adam",
        help="The name of an optimizer."
    )
    parser.add_argument(
        "--optimizer_config",
        default="{}",
        type=json.loads,
    )
    return parser


# def add_scheduler_


def main():
    args = add_optimizer_args(ArgumentParser()).parse_args()
    print(args.optimizer_config)


if __name__ == '__main__':
    main()
