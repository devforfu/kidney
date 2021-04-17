import inspect
import os

import json
import pytorch_lightning as pl

import yaml
from pydantic import BaseModel, Field, validator, BaseSettings
from typing import Dict, Any, Optional, Callable, List, Tuple

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import AttributeDict
from zeus.utils import TimestampFormat

from kidney.cli.lightning import get_trainer_specific_args
from kidney.parameters import PROJECT_NAME


class ExperimentConfig(BaseModel):
    name: str
    output_dir: str
    dataset: str
    run_identifier: str = ""
    file_format: str = "enum"
    timestamp: str = TimestampFormat.VerboseShortSeconds.now()
    tags: List[str] = Field(default_factory=list)
    project_name: str = PROJECT_NAME

    @validator("run_identifier", pre=True, always=True)
    def use_default_run_identifier(cls, value, *, values):
        return value if value else values["name"]

    @validator("output_dir", pre=True, always=True)
    def expand_output_dir_user(cls, value):
        return os.path.expanduser(value)

    @property
    def root_dir(self) -> str:
        return os.path.join(self.output_dir, self.name)


class TrainingConfig(BaseModel):
    loss_name: str
    loss_config: Optional[Dict[str, Any]] = None
    seed: int = 1
    logging_steps: int = 1
    max_epochs: int = 1
    batch_size: int = 64
    num_workers: int = 12
    multiprocessing_context: str = "fork"


class ValidationFolds(BaseModel):
    n_folds: int
    train: List[int]
    valid: List[int]


class ValidationConfig(BaseSettings):
    filepath: str = Field(env="VAL_FILE")
    fold_no: int = Field(default=0, env="VAL_FOLD")
    folds: ValidationFolds = None

    @validator("filepath", pre=True, always=True)
    def filepath_exists(cls, value):
        if not os.path.exists(value):
            raise FileNotFoundError(value)
        return value

    @validator("folds", always=True)
    def read_validation_folds(cls, value, *, values):
        with open(values["filepath"]) as fp:
            return ValidationFolds(**json.load(fp))


class EarlyStoppingConfig(BaseModel):
    monitor: str = "val_loss"
    mode: str = "min"
    patience: int = 1
    restore_best_weights: bool = False
    enabled: bool = True


class CheckpointConfig(BaseModel):
    monitor: str = "val_loss"
    folder: str = "checkpoints"
    mode: str = "min"
    top_k: int = 1
    enabled: bool = True
    filepath: Optional[str] = None


class OptimizerConfig(BaseModel):
    name: str = "adam"
    options: Optional[Dict[str, Any]] = None


class SchedulerConfig(BaseModel):
    name: str = "reduce_lr"
    monitor: str = "val_loss"
    interval: str = "epoch"
    options: Optional[Dict[str, Any]] = None


class AugmentationConfig(BaseModel):
    prob: float = 1.0


class BCG(AugmentationConfig):
    brightness: float = 0.2
    contrast: float = 0.2
    gamma: Tuple[float, float] = (80, 120)


class Blur(AugmentationConfig):
    limit: int = 3


class Noise(AugmentationConfig):
    gauss_var: float = 0.001
    affine_scale: Optional[float] = None


class Flip(AugmentationConfig):
    vertical: bool = True
    horizontal: bool = True


class ShiftScaleRotate(AugmentationConfig):
    shift: Optional[float] = 0.0625
    scale: Optional[float] = 0.1
    rotate: Optional[int] = 45


class Elastic(AugmentationConfig):
    alpha: int = 1
    sigma: int = 50
    alpha_affine: int = 50


class TransformersConfig(BaseModel):
    image_size: int
    bcg: Optional[BCG] = BCG()
    blur: Optional[Blur] = Blur()
    noise: Optional[Noise] = Noise()
    flip: Optional[Flip] = Flip()
    elastic: Optional[Elastic] = Elastic()
    ssr: Optional[ShiftScaleRotate] = ShiftScaleRotate()


class Config(BaseModel):
    experiment: ExperimentConfig
    training: TrainingConfig
    validation: ValidationConfig
    optimizer: OptimizerConfig
    transformers: TransformersConfig
    use_wandb_logging: bool = True
    model: Optional[Dict[str, Any]] = None
    metrics: Optional[List[Dict[str, Any]]] = None
    scheduler: Optional[SchedulerConfig] = None
    early_stopping: Optional[EarlyStoppingConfig] = None
    checkpoint: Optional[CheckpointConfig] = None
    lightning: Optional[Dict[str, Any]] = None


MainFunction = Callable[[Config], None]


def configure(func: MainFunction) -> MainFunction:
    signature = inspect.signature(func)

    assert len(signature.parameters) == 1, "The wrapped function can take one argument"
    assert "config" in signature.parameters, "The argument should be called 'config'"

    filename = os.environ.get("CONFIG_FILE", "configuration.yaml")
    assert os.path.exists(filename), "Configuration file is not found!"

    def wrapper() -> None:
        config_cls = signature.parameters["config"].annotation
        with open(filename) as fp:
            config = config_cls(**yaml.safe_load(fp))
        func(config)

    return wrapper


def create_trainer(config: Config) -> pl.Trainer:
    trainer_config = AttributeDict(
        default_root_dir=config.experiment.root_dir,
        weights_save_path=config.experiment.root_dir,
        logger=WandbLogger(
            save_dir=config.experiment.output_dir,
            name=config.experiment.run_identifier,
            version=config.experiment.timestamp,
            tags=config.experiment.tags,
            project=config.experiment.project_name,
        )
        if config.use_wandb_logging
        else None
    )

    callbacks = []

    if config.early_stopping is not None and config.early_stopping.enabled:
        from kidney.extensions.early_stopping import EarlyStopping
        callbacks.append(EarlyStopping(
            monitor=config.early_stopping.monitor,
            mode=config.early_stopping.mode,
            patience=config.early_stopping.patience,
            restore_best_weights=config.early_stopping.restore_best_weights,
        ))

    if config.checkpoint is not None and config.checkpoint.enabled:
        filepath = config.checkpoint.filepath
        if filepath is None:
            filepath = "%s/%s/%s/%s/{epoch:d}_{%s:4f}" % (
                config.experiment.output_dir,
                config.experiment.name,
                config.checkpoint.folder,
                config.experiment.timestamp,
                config.checkpoint.monitor,
            )
        callbacks.append(ModelCheckpoint(
            filepath=filepath,
            monitor=config.checkpoint.monitor,
            mode=config.checkpoint.mode,
            save_top_k=config.checkpoint.top_k,
        ))
    else:
        trainer_config["checkpoint_callback"] = False

    defaults = get_trainer_specific_args(config.lightning or {})

    final_config = defaults.copy()
    final_config.update(trainer_config)
    final_config["callbacks"] = callbacks

    if "gpus" in final_config and isinstance(final_config["gpus"], int):
        final_config["gpus"] = [final_config["gpus"]]

    return pl.Trainer(**final_config)
