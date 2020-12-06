from typing import Optional, Dict, Callable, List, Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import AttributeDict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ExponentialLR, CosineAnnealingLR  # noqa


class BaseExperiment(pl.LightningModule):  # noqa

    def __init__(self, params: AttributeDict):
        super().__init__()
        self.hparams = params
        self.training_loss = 0.0
        self.logging_loss = 0.0
        self.model_outputs = []
        self.true_targets = []
        self.model = self.create_model()
        self.loss_fn: Optional[Callable] = None
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
        self.loss_fn = create_loss(self.hparams)
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

    def forward(self, batch: Dict) -> torch.Tensor:
        return self.model(batch)

    def training_step(self, batch: Dict, batch_no: int) -> Dict:
        outputs = self(batch)
        loss = self.loss_fn(outputs, batch["targets"])
        self.training_loss += loss.item()
        logging_steps = self.hparams.logging_steps
        if self.global_step % logging_steps == 0:
            self.logger.experiment.log({
                "lr": self.last_learning_rate,
                "loss": (self.training_loss - self.logging_loss) / logging_steps
            })
            self.logging_loss = self.training_loss
        step_metrics = [metric(outputs, batch) for metric in self.metrics]
        return {"loss": loss, "step_metrics": step_metrics}

    def validation_epoch_end(self, outputs: List[Any]) -> Dict:
        return {}

    def create_model(self) -> nn.Module:
        raise NotImplementedError("model builder is not defined")

    @property
    def metrics(self) -> List[Callable]:
        return []


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
