import ast
import os
from collections import defaultdict
from dataclasses import dataclass
from os.path import join
from typing import Optional, Dict, Callable, List, Any, Tuple, Type

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from monai.metrics import DiceMetric, compute_confusion_metric
from pytorch_lightning.utilities import AttributeDict
from segmentation_models_pytorch.losses import BINARY_MODE
from segmentation_models_pytorch.losses.dice import DiceLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ExponentialLR, CosineAnnealingLR, OneCycleLR, \
    ReduceLROnPlateau  # noqa
from zeus.utils import classname

from kidney.models.fcn import create_fcn_model


class BaseExperiment(pl.LightningModule):  # noqa

    def __init__(self, params: AttributeDict):
        super().__init__()
        self.hparams = params
        self.training_loss = 0.0
        self.logging_loss = 0.0
        self.model_outputs = []
        self.true_targets = []
        self.model = self.create_model()
        self.metrics: Optional[List[Callable]] = create_metrics(self.hparams)
        self.loss_fn: Optional[Callable] = create_loss(self.hparams)
        self.scheduler: Optional[_LRScheduler] = None

    def model_parameters(self) -> List:
        return self.model.parameters()  # noqa

    @property
    def last_learning_rate(self) -> float:
        if self.scheduler is None:
            return self.hparams.learning_rate
        elif isinstance(self.scheduler, ReduceLROnPlateau):
            return getattr(self.scheduler, "_last_lr", self.hparams.learning_rate)
        return self.scheduler.get_last_lr()[0]

    def configure_optimizers(self) -> Dict:
        opt = create_optimizer(self.model_parameters(), self.hparams)
        self.scheduler = create_scheduler(opt, self)
        if self.scheduler is None:
            return {"optimizer": opt}
        else:
            config = {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "interval": self.hparams.scheduler_interval
                }
            }
            if isinstance(self.scheduler, ReduceLROnPlateau):
                config["lr_scheduler"]["monitor"] = self.hparams.early_stopping_metric
            return config

    def training_step(self, batch: Dict, batch_no: int) -> Dict:
        outputs = self(batch)
        loss = outputs["loss"]
        self.training_loss += loss.item()
        self._log_optimization_metrics()
        step_metrics = self._log_performance_metrics(outputs, batch)
        return {"loss": loss, **step_metrics}

    def validation_step(self, batch: Dict, batch_no: int) -> Dict:
        outputs = self(batch)
        loss = outputs["loss"]
        step_metrics = self._log_performance_metrics(outputs, batch)
        return {"val_loss": loss, **{f"val_{k}": v for k, v in step_metrics.items()}}

    def test_step(self, batch: Dict, batch_no: int) -> Dict:
        outputs = self(batch)
        loss = outputs["loss"]
        step_metrics = self._log_performance_metrics(outputs, batch)
        return {"test_loss": loss, **{f"test_{k}": v for k, v in step_metrics.items()}}

    def training_epoch_end(self, outputs: List[Any]) -> None:
        if hasattr(self.logger, "experiment"):
            self.logger.experiment.log(
                compute_average_metrics(outputs, suffix="avg_trn_")
            )
            self.logger.experiment.log({"current_epoch": self.current_epoch})

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        avg_val_metrics = compute_average_metrics(outputs, suffix="avg_")
        if hasattr(self.logger, "experiment"):
            self.logger.experiment.log(avg_val_metrics)
        self.log_dict(avg_val_metrics)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        avg_test_metrics = compute_average_metrics(outputs, suffix="avg_")
        self.log_dict(avg_test_metrics)

    def create_model(self) -> nn.Module:
        raise NotImplementedError("model's builder is not defined")

    def forward(self, batch: Dict) -> Dict:
        raise NotImplementedError("model's forward method is not defined")

    def _log_optimization_metrics(self):
        if self.logger is None:
            return
        logging_steps = self.hparams.logging_steps
        if self.global_step % logging_steps == 0:
            self.logger.experiment.log({
                "lr": self.last_learning_rate,
                "loss": (self.training_loss - self.logging_loss) / logging_steps
            })
            self.logging_loss = self.training_loss

    def _log_performance_metrics(self, outputs: Dict, batch: Dict):
        step_metrics = {
            metric.__name__: metric(outputs, batch)
            for metric in self.metrics}
        return step_metrics


class FCNExperiment(BaseExperiment):

    def create_model(self) -> nn.Module:
        return create_fcn_model(self.hparams)

    def forward(self, batch: Dict) -> Dict:
        outputs = self.model(batch["img"])
        predicted_mask = outputs["out"]
        if "seg" in batch:
            loss = self.loss_fn(predicted_mask, batch["seg"])
            if isinstance(loss, Dict):
                loss["outputs"] = predicted_mask
                return loss
            else:
                return {"loss": loss, "outputs": predicted_mask}
        return {"outputs": predicted_mask}


def create_optimizer(optimizer_params: List, hparams: AttributeDict) -> Optimizer:
    name = hparams.optimizer_name
    name_normalized = name.lower().strip()
    config = hparams.optimizer_config
    if name_normalized == "adam":
        opt = torch.optim.AdamW(
            params=optimizer_params,
            lr=hparams.learning_rate,
            betas=config.get("betas", [0.9, 0.999]),
            weight_decay=hparams.weight_decay)
    else:
        raise ValueError(f"unknown optimizer: {name}")
    return opt


def create_scheduler(optimizer: Optimizer, experiment: BaseExperiment) -> _LRScheduler:
    hparams = experiment.hparams
    name = hparams.scheduler_name
    if name is None:
        return None
    name_normalized = name.lower().strip()
    config = hparams.scheduler_config
    if name_normalized == "expo":
        scheduler = ExponentialLR(optimizer, gamma=config.get("scheduler_gamma", 0.8))
    elif name_normalized == "cosine":
        if hparams.scheduler_interval == "epoch":
            t_max = hparams.epochs
        elif hparams.scheduler_interval == "step":
            dl = experiment.train_dataloader()
            t_max = len(dl) * hparams.max_epochs
        else:
            raise ValueError(f"unknown scheduler interval: {hparams.scheduler_interval}")
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=t_max,
            eta_min=config.get("eta_min", 1e-6)
        )
    elif name_normalized == "one_cycle":
        dl = experiment.train_dataloader()
        scheduler = OneCycleLR(
            optimizer=optimizer,
            total_steps=len(dl) * hparams.max_epochs,
            **config
        )
    elif name_normalized == "reduce_lr":
        scheduler = ReduceLROnPlateau(optimizer=optimizer, **config)
    else:
        raise ValueError(f"unknown optimizer: {name}")
    return scheduler


def create_loss(hparams: AttributeDict) -> Callable:
    name_normalized = hparams.loss_name
    if name_normalized == "dice_sigmoid":
        from monai.losses import DiceLoss
        return DiceLoss(sigmoid=True)
    elif name_normalized == "dice_bce_weighted":
        config = hparams.loss_config
        return CombinedDiceBCELoss(**config)
    elif name_normalized == "bce_logits":
        return nn.BCEWithLogitsLoss()
    elif name_normalized == "bce_sigmoid":
        return nn.BCELoss()
    elif name_normalized == "bce_jaccard":
        from segmentation_models_pytorch.losses.jaccard import JaccardLoss
        return JaccardLoss(mode="binary", from_logits=True)
    raise ValueError(f"unknown loss function: {hparams.loss_name}")


def create_metrics(hparams: AttributeDict) -> List[Callable]:
    return (
        [create_metric(name) for name in hparams.metrics]
        if hparams.metrics is not None
        else []
    )


def create_metric(name: str) -> Callable:
    from kidney.cli import parse_callable_definition
    metric, kwargs = parse_callable_definition(name, param_separator=",")
    if metric == "loss":
        if "key" not in kwargs:
            raise ValueError(f"cannot initialize '{name}' metric without 'key' parameter")
        return DictKeyGetter(kwargs["key"])
    elif metric == "dice":
        return DictMetric(DiceMetric(**kwargs), "dice")
    elif metric == "dice_coe_sigmoid":
        return DictMetric(DiceCOESigmoid(**kwargs))
    elif metric == "recall":
        return DictMetric(ConfusionMatrixMetric("recall"))
    elif metric == "precision":
        return DictMetric(ConfusionMatrixMetric("precision"))
    elif metric == "f1_score":
        return DictMetric(ConfusionMatrixMetric("f1 score"))
    elif metric == "accuracy":
        return DictMetric(ConfusionMatrixMetric("accuracy"))
    elif metric == "balanced_accuracy":
        return DictMetric(ConfusionMatrixMetric("balanced accuracy"))
    raise ValueError(f"unknown metric name was requested: {name}")


class DictMetric:
    """Takes values from given dictionaries and forwards them into a metric function.

    A metric function should take two tensors: predicted and true segmentation mask.
    The masks are stored in dictionaries. This facade extracts them and forwards into
    a metric function.
    """
    def __init__(
        self,
        metric: Callable,
        name: Optional[str] = None,
        pred_key: str = "outputs",
        true_key: str = "seg"
    ):
        name = name or getattr(metric, "__name__")
        assert name is not None, "metric name is not available!"
        self.metric = metric
        self.name = name
        self.pred_key = pred_key
        self.true_key = true_key

    @property
    def __name__(self) -> str:
        return self.name

    def __call__(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        pred = outputs[self.pred_key]
        true = batch[self.true_key]
        metric = self.metric(pred, true)
        return metric

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"{classname(self)}({self.metric})"


@dataclass
class DiceCOESigmoid:
    """A flexible dice metric implementation supporting various computation algorithms.

    References
    ----------
    [1] https://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/cost.html#dice_coe
    [2] https://www.kaggle.com/wrrosa/hubmap-tf-with-tpu-efficientunet-512x512-train
    """
    loss_type: str = "jaccard"
    threshold: Optional[float] = None
    axis: Tuple[int, ...] = (1, 2, 3)
    smooth: float = 1e-10

    @property
    def __name__(self) -> str:
        return "dice_coe_sigmoid"

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred = pred.sigmoid()
        if self.threshold is not None:
            pred = (pred > self.threshold).float()
        intersect = torch.sum(pred * gt, dim=self.axis)
        if self.loss_type == "jaccard":
            a = torch.sum(pred * pred, dim=self.axis)
            b = torch.sum(gt * gt, dim=self.axis)
        elif self.loss_type == "sorensen":
            a = torch.sum(pred, dim=self.axis)
            b = torch.sum(gt, dim=self.axis)
        else:
            raise ValueError(f"unknown loss type: {self.loss_type}")
        dice = (2. * intersect + self.smooth)/(a + b + self.smooth)
        dice_avg = torch.mean(dice)
        return dice_avg


@dataclass
class ConfusionMatrixMetric:
    metric_name: str
    sigmoid: bool = True

    @property
    def __name__(self):
        return self.metric_name

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        activation = "sigmoid" if self.sigmoid else None
        metric = compute_confusion_metric(
            pred, gt,
            activation=activation,
            metric_name=self.metric_name
        )
        return metric

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"{classname(self)}(metric=\"{self.metric_name}\", sigmoid={self.sigmoid})"


@dataclass
class DictKeyGetter:
    key: str

    @property
    def __name__(self) -> str:
        return self.key

    def __call__(self, outputs: Dict, _: Dict) -> torch.Tensor:
        return outputs[self.key]


def parse_metric_name(name: str):
    try:
        metric, kwargs = name.split(":")
    except ValueError:
        metric, kwargs = name, {}
    else:
        parsed = {}
        for kv in kwargs.split(","):
            k, v = kv.split("=")
            try:
                v = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                pass
            parsed[k] = v
        kwargs = parsed
    return metric, kwargs


def compute_average_metrics(outputs: List[Any], suffix: Optional[str] = None) -> Dict:
    acc = defaultdict(list)
    for record in outputs:
        for k, v in record.items():
            acc[k].append(v.item())
    return {
        k if suffix is None else f"{suffix}{k}": np.mean(collected)
        for k, collected in acc.items()
    }


def save_experiment_info(
    trainer: pl.Trainer,
    info: Dict,
    filename: str = "info"
) -> Optional[str]:
    try:
        [checkpoints] = [
            cb
            for cb in trainer.callbacks
            if classname(cb) == "ModelCheckpoint"
        ]
    except ValueError:
        return None
    dir_path = checkpoints.dirpath
    os.makedirs(dir_path, exist_ok=True)
    torch.save(info, join(dir_path, f"{filename}.pth"))
    return dir_path


class CombinedDiceBCELoss(nn.Module):

    def __init__(
        self,
        smooth: float = 0.0,
        dice_weight: float = 1.0,
    ):
        assert 0 <= dice_weight <= 1, "dice weight should be in the range [0; 1]"
        super().__init__()
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(mode=BINARY_MODE, smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        dice = self.dice_loss(y_pred, y_true)
        bce = self.bce_loss(y_pred, y_true)
        loss = dice * self.dice_weight + bce * (1 - self.dice_weight)
        return {"loss": loss, "dice": dice, "bce": bce}
