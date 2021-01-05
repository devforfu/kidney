import os
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import AttributeDict
from zeus.utils import TimestampFormat

from kidney.parameters import get_relevant_params, as_attribute_dict, requires


@requires([
    "experiment_dir",
    "experiment_name",
    "run_name",
    "dataset",
    "early_stopping_enabled",
    "early_stopping_metric",
    "early_stopping_mode",
    "early_stopping_patience",
    "checkpoints_enabled",
    "checkpoints_metric",
    "checkpoints_mode",
    "checkpoints_top_k",
    "wandb_logging_enabled",
    "timestamp",
    "tags",
    "project_name"
])
def make_trainer_init_params(params: AttributeDict) -> AttributeDict:
    """Creates a dictionary of parameters to initialize PyTorch Lightning trainer."""

    root = os.path.join(params.experiment_dir, params.experiment_name)
    if "fold" in params:
        root = os.path.join(root, params["fold"])

    if params.timestamp is None:
        params.timestamp = TimestampFormat.VerboseShortSeconds.now()

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
        if params.wandb_logging_enabled
        else None
    )

    callbacks = []

    if params.early_stopping_enabled:
        callbacks.append(EarlyStopping(
            monitor=params.early_stopping_metric,
            mode=params.early_stopping_mode,
            patience=params.early_stopping_patience
        ))

    if params.checkpoints_enabled:
        filepath = params.get('checkpoints_path')
        if filepath is None:
            filepath = '%s/%s/checkpoints/%s/{epoch:d}_{%s:.4f}' % (
                params.experiment_dir,
                params.experiment_name,
                params.timestamp,
                params.checkpoints_metric
            )
        callbacks.append(ModelCheckpoint(
            filepath=filepath,
            monitor=params.checkpoints_metric,
            mode=params.checkpoints_mode,
            save_top_k=params.checkpoints_top_k
        ))
    else:
        config["checkpoint_callback"] = False

    defaults = get_trainer_specific_args(params)

    final_config = AttributeDict(defaults.copy())
    final_config.update(config)  # override defaults with given params
    final_config["callbacks"] = callbacks
    if 'gpus' in final_config and isinstance(final_config['gpus'], int):
        final_config['gpus'] = [final_config['gpus']]

    return final_config


def get_trainer_specific_args(params: AttributeDict) -> AttributeDict:
    """From a dictionary of parameters, takes only the parameters relevant
    for PyTorch Lightning trainer.
    """
    return as_attribute_dict(get_relevant_params(pl.Trainer.__init__, params))
