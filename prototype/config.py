import inspect
import os

import json
import pytorch_lightning as pl

import yaml
from pydantic import BaseModel, Field, validator, BaseSettings
from typing import Dict, Any, Optional, Callable, List, Tuple, Union

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import AttributeDict
from zeus.utils import TimestampFormat

from kidney.cli.lightning import get_trainer_specific_args
from kidney.datasets.kaggle import SampleType
from kidney.datasets.sampled import DeformationConfig
from kidney.parameters import PROJECT_NAME


class ExperimentConfig(BaseSettings):
    name: str
    output_dir: str
    dataset: str
    file_format: str = "enum"
    run_identifier: str = Field(default="", env="RUN_ID")
    tags: List[str] = Field(default_factory=list)
    project_name: str = PROJECT_NAME
    timestamp: str = TimestampFormat.VerboseShortSeconds.now()

    @validator("run_identifier", pre=True, always=True)
    def use_default_run_identifier(cls, value, *, values):
        return value if value else values["name"]

    @validator("output_dir", pre=True, always=True)
    def expand_output_dir_user(cls, value):
        return os.path.expanduser(value)

    @property
    def root_dir(self) -> str:
        root_dir = os.path.join(self.output_dir, self.name)
        if self.run_identifier:
            root_dir = f"{root_dir}_{self.run_identifier}"
        return root_dir


class TrainingConfig(BaseModel):
    loss_name: str
    loss_config: Optional[Dict[str, Any]] = None
    seed: int = 1
    logging_steps: int = 1
    batch_size: int = 64
    num_workers: int = 12
    multiprocessing_context: str = "fork"


class ValidationFolds(BaseModel):
    n_folds: int
    train: Union[List[int], List[List[int]], List[str], List[List[str]]]
    valid: Union[List[int], List[List[int]], List[str], List[List[str]]]


class ValidationConfig(BaseSettings):
    filepath: str = Field(env="VAL_FILE")
    fold_no: int = Field(default=0, env="VAL_FOLD")
    folds: ValidationFolds = None

    def get_selected_fold(self) -> Tuple[List[int], List[int]]:
        return (
            (self.folds.train, self.folds.valid)
            if self.folds.n_folds == 1
            else (
                self.folds.train[self.fold_no],
                self.folds.valid[self.fold_no],
            )
        )

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



class HSV(AugmentationConfig):
    hue: int = 10
    saturation: int = 15
    value: int = 10


class HistogramEqualization(AugmentationConfig):
    clip: Tuple[float, float] = (1, 2)
    tile_grid: Tuple[int, int] = (8, 8)


class BrightnessContrast(AugmentationConfig):
    brightness: float = 0.2
    contrast: float = 0.2


class ColorTransform(AugmentationConfig):
    hsv: Optional[HSV] = HSV()
    hist: Optional[HistogramEqualization] = HistogramEqualization()
    brightness_contrast: Optional[BrightnessContrast] = BrightnessContrast()


class TransformersConfig(BaseModel):
    image_size: int
    resize: bool = True
    normalize_only: bool = False
    bcg: Optional[BCG] = BCG()
    blur: Optional[Blur] = Blur()
    noise: Optional[Noise] = Noise()
    flip: Optional[Flip] = Flip()
    elastic: Optional[Elastic] = Elastic()
    ssr: Optional[ShiftScaleRotate] = ShiftScaleRotate()
    color: Optional[ColorTransform] = ColorTransform()


class Config(BaseModel):
    experiment: ExperimentConfig
    training: TrainingConfig
    validation: ValidationConfig
    optimizer: OptimizerConfig
    transformers: TransformersConfig
    use_wandb_logging: bool = True
    deformation: Optional[DeformationConfig] = None
    model: Optional[Dict[str, Any]] = None
    metrics: Optional[List[Dict[str, Any]]] = None
    scheduler: Optional[SchedulerConfig] = None
    early_stopping: Optional[EarlyStoppingConfig] = None
    checkpoint: Optional[CheckpointConfig] = None
    lightning: Optional[Dict[str, Any]] = None
    dataset: Optional[Dict[str, Any]] = None


class SlidingWindowConfig(BaseModel):
    window_size: int = 1024
    overlap: int = 32
    max_batch_size: int = 32
    check_for_outliers: bool = True
    outliers_threshold: int = 1000
    transform_input: Optional[Callable] = None
    transform_output: Optional[Callable] = None


class PredictConfig(BaseSettings):
    experiment_id: str
    factory_class: str
    dataset: str
    output_dir: str
    sliding_window: SlidingWindowConfig
    run_id: str = Field(env="RUN_ID")
    checkpoints_root: str = os.path.expanduser("~/experiments")
    sample_type: SampleType = SampleType.All
    device: str = Field(default="cuda:0", env="DEVICE")
    performance_metric: str = "avg_val_dice"
    keys_include: Optional[List[str]] = None
    encode_masks: bool = True
    debug: bool = False

    @property
    def root_dir(self) -> str:
        return os.path.join(self.checkpoints_root, self.experiment_id, "checkpoints")

    @property
    def storage_dir(self) -> str:
        return os.path.join(self.root_dir, self.run_id)

    @property
    def predictions_file(self) -> str:
        return os.path.join(self.output_dir, self.experiment_id, f"fold_{self.run_id}.csv")


MainFunction = Callable[[Config], None]


def configure(func: MainFunction) -> MainFunction:
    signature = inspect.signature(func)

    assert len(signature.parameters) == 1, "The wrapped function can take one argument"
    assert "config" in signature.parameters, "The argument should be called 'config'"

    filename = os.environ.get("CONFIG_FILE", "configuration.yaml")
    filename = os.path.expanduser(filename)
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
            filepath = "%s/%s/%s/{epoch:d}_{%s:4f}" % (
                config.experiment.root_dir,
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
