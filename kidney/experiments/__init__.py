from collections import defaultdict
from os.path import join
from typing import Optional, Dict, Callable, List, Any

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import AttributeDict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ExponentialLR, CosineAnnealingLR  # noqa
from zeus.utils import classname


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
        return (
            self.hparams.learning_rate
            if self.scheduler is None
            else self.scheduler.get_last_lr()[0]
        )

    def configure_optimizers(self) -> Dict:
        opt = create_optimizer(self.model_parameters(), self.hparams)
        self.scheduler = create_scheduler(opt, self)
        return (
            {"optimizer": opt}
            if self.scheduler is None
            else
            {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "interval": self.hparams.scheduler_interval
                }
            }
        )

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

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.logger.experiment.log(
            compute_average_metrics(outputs, suffix="avg_trn_")
        )

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        avg_val_metrics = compute_average_metrics(outputs, suffix="avg_")
        self.logger.experiment.log(avg_val_metrics)
        self.log_dict(avg_val_metrics)

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
            t_max = len(dl) * hparams.epochs
        else:
            raise ValueError(f"unknown scheduler interval: {hparams.scheduler_interval}")
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=t_max,
            eta_min=config.get("eta_min", 1e-6)
        )
    else:
        raise ValueError(f"unknown optimizer: {name}")
    return scheduler


def create_loss(hparams: AttributeDict) -> Callable:
    name_normalized = hparams.loss_name
    if name_normalized == "dice_sigmoid":
        from monai.losses import DiceLoss
        return DiceLoss(sigmoid=True)
    raise ValueError(f"unknown loss function: {hparams.loss_name}")


def create_metrics(hparams: AttributeDict) -> List[Callable]:
    return []


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
    torch.save(info, join(dir_path, f"{filename}.pth"))
    return dir_path
