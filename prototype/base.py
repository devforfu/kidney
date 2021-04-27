from typing import Dict, Any, Optional, Callable, List, Union

import pytorch_lightning as pl
import torch
from madgrad import MADGRAD
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from segmentation_models_pytorch.losses.jaccard import JaccardLoss
from segmentation_models_pytorch.utils.metrics import IoU
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ExponentialLR,
    CosineAnnealingLR,
    OneCycleLR,
    ReduceLROnPlateau,
)

from kidney.experiments import (
    DictKeyGetter,
    DictMetric,
    DiceCOESigmoid,
    ConfusionMatrixMetric,
    CombinedDiceBCELoss,
    compute_average_metrics,
    SorensenDice,
)
from prototype.config import (
    Config,
    TrainingConfig,
    OptimizerConfig,
)


class Prototype(pl.LightningModule):
    def __init__(self, params: Config):
        super().__init__()
        self.config = params
        self.hparams = params.dict()
        self.training_loss = 0.0
        self.logging_loss = 0.0
        self.model = self.create_model()
        self.loss_fn = create_loss(params.training)
        self.metrics = create_metrics(params.metrics)

    def model_parameters(self) -> List:
        return self.model.parameters()

    def model_weights(self) -> Dict:
        return self.model.state_dict()

    def load_weights(self, state_dict: Dict):
        self.model.load_state_dict(state_dict)

    @property
    def current_learning_rates(self) -> List[float]:
        return [group["lr"] for group in self.optimizers().param_groups]

    @property
    def num_bad_epochs(self) -> int:
        [config] = self.trainer.lr_schedulers
        scheduler = config["scheduler"]
        return getattr(scheduler, "num_bad_epochs", -1)

    def configure_optimizers(self) -> Dict:
        opt = create_optimizer(self.model_parameters(), self.config.optimizer)
        if self.config.scheduler is None:
            return {"optimizer": opt}
        else:
            scheduler = create_scheduler(opt, self)
            config = {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.config.scheduler.interval,
                },
            }
            if isinstance(scheduler, ReduceLROnPlateau):
                config["lr_scheduler"]["monitor"] = self.config.scheduler.monitor
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
            if self.num_bad_epochs != -1:
                self.logger.experiment.log({"num_bad_epochs": self.num_bad_epochs})

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
        logging_steps = self.config.training.logging_steps
        if self.global_step % logging_steps == 0:
            [lr] = self.current_learning_rates
            self.logger.experiment.log(
                {
                    "lr": lr,
                    "loss": (self.training_loss - self.logging_loss) / logging_steps,
                }
            )
            self.logging_loss = self.training_loss

    def _log_performance_metrics(self, outputs: Dict, batch: Dict):
        step_metrics = {
            metric.__name__: metric(outputs, batch) for metric in self.metrics
        }
        return step_metrics


def create_loss(config: TrainingConfig) -> Callable:
    name = config.loss_name.lower()
    if name == "dice_sigmoid":
        return DiceLoss(sigmoid=True)
    elif name == "dice_bce_weighted":
        return CombinedDiceBCELoss(**(config.loss_config or {}))
    elif name == "bce_logits":
        return nn.BCEWithLogitsLoss()
    elif name == "bce_sigmoid":
        return nn.BCELoss()
    elif name == "bce_jaccard":
        return JaccardLoss(mode="binary", from_logits=True)
    raise ValueError(f"unknown loss function: {name}")


def create_metrics(metrics: Optional[Dict[str, Any]] = None):
    return (
        [create_metric(params) for params in metrics]
        if metrics is not None
        else []
    )


def create_metric(params: Dict[str, Any]):
    name = params.pop("name", "").lower()
    if not name:
        raise ValueError("cannot parse metric without name:", params)
    if name == "loss":
        if "key" not in params:
            raise ValueError(
                f"cannot initialize '{name}' metric without 'key' parameter"
            )
        return DictKeyGetter(params["key"])
    elif name == "dice":
        return DictMetric(DiceMetric(**params), "dice")
    elif name == "dice_coe_sigmoid":
        return DictMetric(DiceCOESigmoid(**params))
    elif name == "recall":
        return DictMetric(ConfusionMatrixMetric("recall"))
    elif name == "precision":
        return DictMetric(ConfusionMatrixMetric("precision"))
    elif name == "f1_score":
        return DictMetric(ConfusionMatrixMetric("f1 score"))
    elif name == "accuracy":
        return DictMetric(ConfusionMatrixMetric("accuracy"))
    elif name == "balanced_accuracy":
        return DictMetric(ConfusionMatrixMetric("balanced accuracy"))
    elif name == "iou":
        t = params.get("t")
        name = name if t is None else f"iou_{t:2.2%}"
        return DictMetric(IoU(threshold=t), name=name)
    elif name == "sorensen_dice":
        return DictMetric(SorensenDice(**params))
    raise ValueError(f"unknown metric name was requested: {name}")


def create_optimizer(optimizer_params: List, config: OptimizerConfig) -> Optimizer:
    name = config.name.strip()
    options = config.options or {}
    if name == "adam":
        opt = torch.optim.AdamW(params=optimizer_params, **options)
    elif name == "adam_v":
        opt = torch.optim.Adam(params=optimizer_params, **options)
    elif name == "sgd":
        opt = torch.optim.SGD(params=optimizer_params, **options)
    elif name == "madgrad":
        opt = MADGRAD(params=optimizer_params, **options)
    else:
        raise ValueError(f"unknown optimizer: {name}")
    return opt


Scheduler = Union[
    ExponentialLR, CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau,
]


def create_scheduler(optimizer: Optimizer, experiment: Prototype) -> Scheduler:
    scheduler_config = experiment.config.scheduler
    lightning_config = experiment.config.lightning

    name = scheduler_config.name.lower()
    options = scheduler_config.options or {}

    if name == "expo":
        scheduler = ExponentialLR(optimizer, **options)

    elif name == "cosine":
        if scheduler_config.interval == "epoch":
            t_max = lightning_config["max_epochs"]
        elif scheduler_config.interval == "step":
            dl = experiment.train_dataloader()
            t_max = len(dl) * lightning_config["max_epochs"]
        else:
            raise ValueError(f"unknown scheduler interval: {scheduler_config.interval}")
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=t_max, **options)

    elif name == "one_cycle":
        dl = experiment.train_dataloader()
        total_steps = len(dl) * lightning_config["max_epochs"]
        scheduler = OneCycleLR(optimizer=optimizer, total_steps=total_steps, **options)

    elif name == "reduce_lr":
        scheduler = ReduceLROnPlateau(optimizer=optimizer, **options)

    else:
        raise ValueError(f"unknown optimizer: {name}")

    return scheduler
