import os
from datetime import datetime
from functools import wraps
from typing import List, Callable

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import AttributeDict
from zeus.utils import TimestampFormat

from kidney.parameters import get_relevant_params, as_attribute_dict


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


@requires([
    "experiment_dir",
    "experiment_name",
    "timestamp",
    "run_name",
    "dataset",
    "tags",
    "early_stopping_enabled",
    "early_stopping_metric",
    "early_stopping_mode",
    "early_stopping_patience",
    "checkpoints_enabled",
    "checkpoints_metric",
    "checkpoints_mode",
    "checkpoints_top_k",
])
def make_trainer_init_params(params: AttributeDict) -> AttributeDict:
    """Creates a dictionary of parameters to initialize PyTorch Lightning trainer."""

    params = as_attribute_dict(params)

    root = os.path.join(params.experiment_dir, params.experiment_name)

    if params.timestamp is None:
        params.timestamp = datetime.utcnow().strftime(
            TimestampFormat.VerboseShortSeconds
        )

    if params.run_name is None:
        params.run_name = params.experiment_name

    config = AttributeDict(
        default_root_dir=root,
        weights_save_path=root,
        logger=WandbLogger(
            save_dir=params.experiment_dir,
            name=params.run_name,
            version=params.timestamp,
            tags=params.tags,
            project=params.project_name
        )
        if params.wandb_logger_enabled
        else None
    )

    if params.early_stopping_enabled:
        config["early_stopping_callback"] = EarlyStopping(
            monitor=params.early_stopping_metric,
            mode=params.early_stopping_mode,
            patience=params.early_stopping_patience
        )

    if params.checkpoints_enabled:
        config["checkpoints_callback"] = ModelCheckpoint(

        )

    return AttributeDict({})  # todo: finalize the implementation


def get_trainer_specific_args(params: AttributeDict) -> AttributeDict:
    """From a dictionary of parameters, takes only the parameters relevant
    for PyTorch Lightning trainer.
    """
    return as_attribute_dict(get_relevant_params(pl.Trainer.__init__, params))
