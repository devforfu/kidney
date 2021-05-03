from abc import ABC
from typing import Dict

import segmentation_models_pytorch as smp
from torch import nn

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
