import inspect
from argparse import ArgumentParser

from pydantic import BaseModel
from typing import Callable, Type

MainFunction = Callable[[BaseModel], None]


def cli(func: MainFunction) -> Callable[[], None]:
    signature = inspect.signature(func)

    assert len(signature.parameters) == 1, "The wrapped function can take one argument"
    assert "params" in signature.parameters, "The argument should be called 'params'"

    def wrapper() -> None:
        param = signature.parameters["params"]
        parsed = make_cli(param.annotation).parse_args()
        parsed_config = param.annotation(**vars(parsed))
        return func(parsed_config)

    return wrapper


def make_cli(config_type: Type[BaseModel]) -> ArgumentParser:
    parser = ArgumentParser()
    for field in config_type.__fields__.values():
        parser.add_argument(
            f"--{field.name}",
            type=field.type_,
            required=field.required,
            default=field.default,
            help=field.field_info.description
        )
    return parser

