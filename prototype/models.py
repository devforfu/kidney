from abc import ABC
from typing import Dict

import segmentation_models_pytorch as smp
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits

from prototype.base import Prototype


class SegmentationExperiment(Prototype, ABC):

    def forward(self, batch: Dict) -> Dict:
        predicted_mask = self.model(batch["img"])
        if "seg" in batch:
            loss = self.loss_fn(predicted_mask, batch["seg"])
            if isinstance(loss, Dict):
                loss["outputs"] = predicted_mask
                return loss
            else:
                return {"loss": loss, "outputs": predicted_mask}
        return {"outputs": predicted_mask}


class UppExperiment(SegmentationExperiment):

    def create_model(self) -> nn.Module:
        return smp.UnetPlusPlus(**self.config.model)

    def on_train_epoch_end(self, outputs) -> None:
        from kidney.datasets.sampled import RandomTilesDataset
        dl = self.train_dataloader()
        if isinstance(dl.dataset, RandomTilesDataset):
            dl.dataset.update_deformation()


class LinkNetExperiment(SegmentationExperiment):

    def create_model(self) -> nn.Module:
        return smp.Linknet(**self.config.model)

    def on_train_epoch_end(self, outputs) -> None:
        from kidney.datasets.sampled import RandomTilesDataset
        dl = self.train_dataloader()
        if isinstance(dl.dataset, RandomTilesDataset):
            dl.dataset.update_deformation()


class UnetExperiment(SegmentationExperiment):

    def create_model(self) -> nn.Module:
        return smp.Unet(**self.config.model)

    def on_train_epoch_end(self, outputs) -> None:
        from kidney.datasets.sampled import RandomTilesDataset
        dl = self.train_dataloader()
        if isinstance(dl.dataset, RandomTilesDataset):
            dl.dataset.update_deformation()


class DeepLapV3Plus(SegmentationExperiment):

    def create_model(self) -> nn.Module:
        return smp.DeepLabV3Plus(**self.config.model)

    def on_train_epoch_end(self, outputs) -> None:
        from kidney.datasets.sampled import RandomTilesDataset
        dl = self.train_dataloader()
        if isinstance(dl.dataset, RandomTilesDataset):
            dl.dataset.update_deformation()


class PseudoLabeledExperiment(Prototype, ABC):

    def create_model(self) -> nn.Module:
        model_name = self.config.experiment.name.replace("pseudo_", "")
        if model_name.startswith("deeplab_v3_plus"):
            return smp.DeepLabV3Plus(**self.config.model)
        raise ValueError(f"unknown model: {model_name}")

    def forward(self, batch: Dict) -> Dict:
        predicted_mask = self.model(batch["img"])
        if "seg" in batch:
            loss = self.loss_fn(
                predicted_mask,
                batch["seg"],
                reduction="none",
            )
            if "weight" in batch:
                ws = batch["weight"].float()
                ws = ws[(...,) + (None,)*3]
                loss *= ws
            loss = loss.mean()
            if isinstance(loss, Dict):
                loss["outputs"] = predicted_mask
            else:
                return {"loss": loss, "outputs": predicted_mask}
        return {"outputs": predicted_mask}

    def on_train_epoch_end(self, outputs) -> None:
        from kidney.datasets.sampled import RandomTilesDataset
        dl = self.train_dataloader()
        if isinstance(dl.dataset, RandomTilesDataset):
            dl.dataset.update_deformation()
