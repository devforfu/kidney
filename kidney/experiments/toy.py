"""A toy experiment using randomly generated images and MONAI models."""
import os
from os.path import join
from typing import Tuple, Dict

import PIL.Image
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
from monai.data import create_test_image_2d, Dataset, list_data_collate, PILReader
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ToTensord,
    Activations,
    AsDiscrete
)
from pytorch_lightning.utilities import AttributeDict
from torch.utils.data import DataLoader
from zeus.core.random import super_seed

from kidney.cli import default_args, entry_point
from kidney.cli.basic import basic_parser
from kidney.cli.lightning import make_trainer_init_params
from kidney.cli.models import add_unet_args
from kidney.experiments import BaseExperiment
from kidney.log import get_logger
from kidney.models.unet import create_unet_model


@entry_point(
    base_parser_factory=lambda: basic_parser(__file__),
    extensions=default_args() + [add_unet_args]
)
def main(params: AttributeDict):
    super_seed(params.seed)

    logger = get_logger(__file__)

    logger.info("generating synthetic data")
    trn_files, val_files = generate_synthetic_data(40, "/tmp/generated", 20)

    logger.info("create transformations")
    trn_transformers, val_transformers, post_transformers = create_transformers("img", "seg")

    logger.info("creating data loaders")
    trn_loader, val_loader = create_data_loaders(
        files=(trn_files, val_files),
        transformers=(trn_transformers, val_transformers),
        num_workers=params.num_workers
    )

    trainer = pl.Trainer(**make_trainer_init_params(params))

    trainer.fit(model=ToyExperiment(params),
                train_dataloader=trn_loader,
                val_dataloaders=val_loader)


def generate_synthetic_data(n: int, output_dir: str, train_size: int):
    os.makedirs(output_dir, exist_ok=True)
    image_paths, mask_paths = [], []

    for i in range(n):
        image, mask = create_test_image_2d(128, 128, num_seg_classes=1)

        image_path = join(output_dir, f"img{i:d}.png")
        image_paths.append(image_path)
        PIL.Image.fromarray(image.astype(np.uint8)).save(image_path)

        mask_path = join(output_dir, f"seg{i:d}.png")
        mask_paths.append(mask_path)
        PIL.Image.fromarray(mask.astype(np.uint8)).save(mask_path)

    trn_files = [
        {"img": img, "seg": seg}
        for img, seg in zip(image_paths[:train_size], mask_paths[:train_size])]

    val_files = [
        {"img": img, "seg": seg}
        for img, seg in zip(image_paths[-train_size:], mask_paths[-train_size:])]

    return trn_files, val_files


def create_transformers(image_key: str, mask_key: str) -> Tuple:
    keys = [image_key, mask_key]
    trn_transformers = Compose([
        LoadImaged(reader=PILReader(), keys=keys),
        AddChanneld(keys=keys),
        ScaleIntensityd(keys=image_key),
        RandCropByPosNegLabeld(
            keys=keys,
            label_key=mask_key,
            spatial_size=[96, 96],
            pos=1,
            neg=1,
            num_samples=4
        ),
        RandRotate90d(keys=keys, prob=0.5, spatial_axes=(0, 1)),
        ToTensord(keys=keys)
    ])
    val_transformers = Compose([
        LoadImaged(reader=PILReader(), keys=keys),
        AddChanneld(keys=keys),
        ScaleIntensityd(keys=image_key),
        ToTensord(keys=keys)
    ])
    post_transformers = Compose([
        Activations(sigmoid=True),
        AsDiscrete(threshold_values=True)
    ])
    return trn_transformers, val_transformers, post_transformers


def create_data_loaders(files: Tuple, transformers: Tuple, num_workers: int):
    trn_files, val_files = files
    trn_transformers, val_transformers = transformers

    trn_dataset = Dataset(data=trn_files, transform=trn_transformers)
    trn_loader = DataLoader(
        dataset=trn_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=list_data_collate
    )

    val_dataset = Dataset(data=val_files, transform=val_transformers)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=list_data_collate
    )

    return trn_loader, val_loader


class ToyExperiment(BaseExperiment):

    def create_model(self) -> nn.Module:
        # https://github.com/devforfu/birds/blob/95cc5fbce21dfd4a971753cacb38f05c9ed438d3/birds/experiments/common.py#L146
        # https://github.com/Project-MONAI/tutorials/blob/master/2d_segmentation/torch/unet_training_dict.py
        return create_unet_model(self.hparams)

    def forward(self, batch: Dict) -> Dict:
        predicted_mask = self.model(batch["img"])
        loss = self.loss_fn(predicted_mask, batch["seg"])
        return {"loss": loss, "outputs": predicted_mask}


if __name__ == '__main__':
    main()
