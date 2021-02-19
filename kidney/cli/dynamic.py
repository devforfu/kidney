from argparse import ArgumentParser
from typing import List, Dict

from kidney.cli import extend_parser


def add_dynamically_defined_params(config: List[Dict]) -> ArgumentParser:
    """Extends CLI with arbitrary parameters.

    Helpful for handling cases that aren't covered with other parsers, or that
    are too specific to implement them with a dedicated function.
    """

    @extend_parser
    def extended(parser: ArgumentParser) -> ArgumentParser:
        for param_def in config:
            name = param_def.pop("name")
            if name is None:
                raise RuntimeError(f"invalid param def: {param_def}")
            parser.add_argument(name, **param_def)
        return parser

    return extended
