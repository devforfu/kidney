import inspect
from argparse import Namespace
from functools import wraps
from types import SimpleNamespace
from typing import Callable, Dict, Union, List

from pytorch_lightning.utilities import AttributeDict

PROJECT_NAME = "kidney"


def get_relevant_params(factory: Callable, params_dict: Dict) -> Dict:
    """Extracts a subset of parameters from a dictionary that are present in
    the factory's signature.

    In case if the parameters dictionary includes more parameters than factory
    expects, this function provides a convenience utility to keep only relevant
    parameters when initializing objects instead of passing everything available
    and (probably) get an exception in case if **kwargs parameter is missing
    from a callable's signature.

    Args:
         factory: Object's factory.
         params_dict: Dictionary with parameters where some of the keys aren't
            relevant for a given factory.

    Returns:
        dict: The dictionary with relevant parameters only.

    """
    sig = inspect.signature(factory)
    relevant_args = {
        k: v
        for k, v in params_dict.items()
        if k in sig.parameters
    }
    return relevant_args


def as_attribute_dict(
    params: Union[Namespace, SimpleNamespace, AttributeDict, Dict]
) -> AttributeDict:
    """Converts a set of parameters configuring the experiment into attribute dictionary.

    There are many ways to define parameters as a container with dot-accessed options.
    This function tries to coerce them all to the same type.
    """
    if isinstance(params, (Namespace, SimpleNamespace)):
        dictionary = AttributeDict(vars(params))
    elif isinstance(params, AttributeDict):
        dictionary = params
    elif isinstance(params, Dict):
        dictionary = AttributeDict(params)
    else:
        raise ValueError(f"parameters type unknown: {type(params)}")
    return dictionary


def requires(attributes: List[str]):

    def wrapped(func: Callable):

        @wraps(func)
        def wrapper(params: AttributeDict, **kwargs):
            for attr in attributes:
                if attr not in params:
                    raise ValueError(
                        f"required parameter is missing: {attr}; make sure that your "
                        f"dictionary includes all parameters required by a function."
                    )
            return func(params, **kwargs)

        return wrapper

    return wrapped
