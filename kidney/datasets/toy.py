"""A toy dataset from MONAI example.

References:
    1) https://github.com/Project-MONAI/tutorials/blob/master/2d_segmentation/torch/unet_training_dict.py

"""
import os
import shutil
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from os.path import join
from typing import List, Dict, Tuple

import PIL.Image
import numpy as np
from monai.data import create_test_image_2d, PILReader, Dataset, list_data_collate
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
from torch.utils.data import DataLoader


@dataclass
class ToyData:
    image_size: int
    num_seg_classes: int
    keys: Tuple[str, str]
    train: List[Dict]
    valid: List[Dict]


def generate_synthetic_data(
    total_samples: int,
    image_size: int = 128,
    num_seg_classes: int = 1,
    output_dir: str = "/tmp/toy",
    train_size: float = 0.8,
    image_key: str = "img",
    mask_key: str = "seg",
    rescale: bool = False
) -> ToyData:
    shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    num_train_samples = int(total_samples*train_size)
    num_valid_samples = total_samples - num_train_samples
    records = defaultdict(list)

    for subset, n in (
        ("train", num_train_samples),
        ("valid", num_valid_samples)
    ):
        for i in range(n):
            image, mask = create_test_image_2d(
                width=image_size,
                height=image_size,
                num_seg_classes=num_seg_classes
            )
            record = {}
            for suffix, arr in ((image_key, image), (mask_key, mask)):
                path = join(output_dir, f"{suffix}{i:d}_{subset}.png")
                if rescale:
                    arr *= 255
                PIL.Image.fromarray(arr.astype(np.uint8)).save(path)
                record[suffix] = path
            records[subset].append(record)

    return ToyData(
        image_size=image_size,
        num_seg_classes=num_seg_classes,
        keys=(image_key, mask_key),
        train=records["train"],
        valid=records["valid"]
    )


@dataclass
class Transformers:
    train: Compose
    valid: Compose
    post: Compose


def create_transformers(toy_data: ToyData) -> Transformers:
    image_key, mask_key = keys = toy_data.keys
    crop_size = int(0.75 * toy_data.image_size)
    return Transformers(
        train=Compose([
            LoadImaged(reader=PILReader(), keys=keys),
            AddChanneld(keys=keys),
            ScaleIntensityd(keys=image_key),
            RandCropByPosNegLabeld(
                keys=keys,
                label_key=mask_key,
                spatial_size=[crop_size, crop_size],
                pos=1,
                neg=1,
                num_samples=4
            ),
            RandRotate90d(keys=keys, prob=0.5, spatial_axes=(0, 1)),
            ToTensord(keys=keys)
        ]),
        valid=Compose([
            LoadImaged(reader=PILReader(), keys=keys),
            AddChanneld(keys=keys),
            ScaleIntensityd(keys=image_key),
            ToTensord(keys=keys)
        ]),
        post=Compose([
            Activations(sigmoid=True),
            AsDiscrete(threshold_values=True)
        ])
    )


def create_data_loaders(
    toy_data: ToyData,
    transformers: Transformers,
    batch_sizes: Tuple[int, int] = (2, 1),
    num_workers: int = 0,
) -> OrderedDict:
    assert len(batch_sizes) == 2, "should provide batch sizes for both train and valid loaders"
    train_size, valid_size = batch_sizes
    loaders = OrderedDict()
    for subset, data, transform, batch_size in (
        ("train", toy_data.train, transformers.train, train_size),
        ("valid", toy_data.valid, transformers.valid, valid_size)
    ):
        loaders[subset] = DataLoader(
            dataset=Dataset(
                data=getattr(toy_data, subset),
                transform=getattr(transformers, subset)
            ),
            batch_size=batch_size,
            shuffle=subset == "train",
            num_workers=num_workers,
            collate_fn=list_data_collate
        )
    return loaders
