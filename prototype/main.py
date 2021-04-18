from collections import OrderedDict
from typing import List, Dict

from torch.utils.data.dataloader import DataLoader
from zeus.core.random import super_seed

from kidney.datasets.offline import OfflineCroppedDatasetV2
from kidney.datasets.transformers import Transformers
from kidney.datasets.utils import read_segmentation_info
from kidney.experiments import save_experiment_info
from prototype.config import Config, configure, create_trainer
from prototype.models import UppExperiment
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

    save_experiment_info(trainer, {"params": config, "transformers": transformers})

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



if __name__ == "__main__":
    main()
