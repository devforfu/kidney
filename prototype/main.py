import os
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

from torch.utils.data.dataloader import DataLoader
from zeus.core.random import super_seed

from kidney.datasets.offline import OfflineCroppedDatasetV2
from kidney.datasets.sampled import RandomTilesDataset, TileDataset
from kidney.datasets.transformers import Transformers, IntensityNormalization
from kidney.datasets.utils import read_segmentation_info
from kidney.experiments import save_experiment_info
from prototype.config import Config, configure, create_trainer, ValidationConfig
from prototype.models import UppExperiment
from prototype.transformers import create_transformers


@configure
def main(config: Config):
    super_seed(config.training.seed)

    filename = os.environ.get("CONFIG_FILE", "configuration.yaml")

    tags = config.experiment.tags.copy()
    tags.append(f"run_id:{config.experiment.run_identifier}")
    tags.append(f"fold:{config.validation.fold_no}")
    tags.append(f"config:{Path(filename).stem}")
    config.experiment.tags = tags

    train, valid = config.validation.get_selected_fold()

    if config.experiment.file_format in ("enum", "bbox"):
        samples = read_segmentation_info(
            folder=config.experiment.dataset,
            file_format=config.experiment.file_format
        )
        transformers = create_transformers(config.transformers),
        loaders = create_data_loaders(
            samples=samples,
            train=train,
            valid=valid,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            transformers=transformers,
            multiprocessing_context=config.training.multiprocessing_context,
        )

    elif config.experiment.file_format == "sampling":
        transformers = create_transformers(
            config=config.transformers,
            normalization=IntensityNormalization.DatasetScale2,
        )
        loaders = create_data_loaders_tiled(
            path=config.experiment.dataset,
            train=train,
            valid=valid,
            image_size=config.transformers.image_size,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            transformers=transformers,
            multiprocessing_context=config.training.multiprocessing_context,
            dataset_options=config.dataset,
        )

    else:
        raise RuntimeError(f"wrong configuration: {config.experiment.file_format}")

    trainer = create_trainer(config)

    save_experiment_info(trainer, {"params": config, "transformers": transformers})

    trainer.fit(
        model=UppExperiment(config),
        train_dataloader=loaders["train"],
        val_dataloaders=loaders.get("valid"),
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


def create_data_loaders_tiled(
    path: str,
    train: List[str],
    valid: List[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
    transformers: Transformers,
    dataset_options: Optional[Dict[str, Any]] = None,
    **dataloader_options,
) -> OrderedDict:

    train_data_opts, valid_data_opts = {}, {}
    if dataset_options is not None:
        train_data_opts.update(dataset_options.get("train", {}))
        valid_data_opts.update(dataset_options.get("valid", {}))

    tile_shape = tuple(dataset_options.get("tile_shape", (image_size, image_size)))

    if "tile_shape" not in train_data_opts:
        train_data_opts["tile_shape"] = tile_shape

    if "tile_shape" not in valid_data_opts:
        valid_data_opts["tile_shape"] = tile_shape

    loaders = OrderedDict([
        ("train", DataLoader(
            dataset=RandomTilesDataset(
                path=path,
                keys=train,
                transform=transformers.train,
                **train_data_opts,
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            **dataloader_options
        )),
    ])
    if valid:
        loaders["valid"] = DataLoader(
            dataset=TileDataset(
                path=path,
                keys=valid,
                transform=transformers.valid,
                **valid_data_opts,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            **dataloader_options
        )
    return loaders


if __name__ == "__main__":
    main()
