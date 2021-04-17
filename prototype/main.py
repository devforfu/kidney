from abc import ABC
from collections import OrderedDict
from typing import List, Dict

import segmentation_models_pytorch as smp
from torch import nn
from torch.utils.data.dataloader import DataLoader
from zeus.core.random import super_seed

from kidney.datasets.offline import OfflineCroppedDatasetV2
from kidney.datasets.transformers import Transformers
from kidney.datasets.utils import read_segmentation_info
from prototype.base import Prototype
from prototype.config import Config, configure, create_trainer
from prototype.transformers import create_transformers


@configure
def main(config: Config):
    super_seed(config.training.seed)

    samples = read_segmentation_info(config.experiment.dataset, config.experiment.file_format)

    transformers = create_transformers(config.transformers)

    loaders = create_data_loaders(
        samples=samples,
        train=config.validation.folds.train,
        valid=config.validation.folds.valid,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        transformers=transformers,
        multiprocessing_context=config.training.multiprocessing_context,
    )

    trainer = create_trainer(config)

    trainer.fit(
        model=UppExperiment(config),
        train_dataloader=loaders["train"],
        val_dataloaders=loaders["valid"],
    )


def create_data_loaders(
    samples: Dict,
    train: List[str],
    valid: List[str],
    batch_size: int,
    num_workers: int,
    transformers: Transformers,
    **dataloader_options,
) -> OrderedDict:
    loaders = OrderedDict()
    for subset, indexes, transform in (
        ("train", train, transformers.train),
        ("valid", valid, transformers.valid)
    ):
        subset_of_samples = [x for i in indexes for x in samples if x["index"] == i]
        loaders[subset] = DataLoader(
            dataset=OfflineCroppedDatasetV2(
                samples=subset_of_samples,
                transform=transform,
            ),
            batch_size=batch_size,
            shuffle=subset == "train",
            num_workers=num_workers,
            **dataloader_options
        )
    return loaders


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


if __name__ == "__main__":
    main()
