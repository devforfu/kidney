from dataclasses import dataclass
from typing import Union, Tuple

from monai.data import PILReader
from monai.transforms import (
    Compose, RandCropByPosNegLabeld, RandSpatialCropSamplesd,
    LoadImaged, ScaleIntensityd, RandRotate90d, ToTensord,
    Activations, AsDiscrete, AsChannelFirstd, AddChanneld
)


@dataclass
class Transformers:
    train: Compose
    valid: Compose
    post: Compose


def create_transformers_crop_to_many(
    image_key: str,
    mask_key: str,
    image_size: Union[int, Tuple[int, int]],
    crop_fraction: float = 0.5,
    num_samples: int = 4,
    rotation_prob: float = 0.5,
    crop_balanced: bool = True,
) -> Transformers:
    """Created transformers with default transformation scheme from MONAI example.

    Parameters
    ----------
    image_key
        Image key.
    mask_key
        Mask key.
    image_size
        Training image and mask size.
    crop_fraction
        Transformation crop fraction.
    num_samples
        Number of crops to generate from a single sample.
    rotation_prob
        Rotation transformation probability.
    crop_balanced
        If True, crop sample image while keeping balance between negative and positive
        classes. Doesn't work properly in case if some samples of the dataset don't
        have segmentation mask.

    Returns
    -------
    transformers
        The structure with transformers.

    """
    if isinstance(image_size, tuple):
        h, w = image_size
    else:
        h = w = image_size
    keys = image_key, mask_key
    spatial_size = [int(h*crop_fraction), int(w*crop_fraction)]

    random_crop = (
        RandCropByPosNegLabeld(
            keys=keys, label_key=mask_key,
            spatial_size=spatial_size,
            pos=1, neg=1, num_samples=num_samples
        )
        if crop_balanced
        else
        RandSpatialCropSamplesd(
            keys=keys, roi_size=spatial_size,
            num_samples=num_samples, random_size=False
        )
    )

    return Transformers(
        train=Compose([
            LoadImaged(reader=PILReader(), keys=keys),
            AsChannelFirstd(keys=image_key),
            AddChanneld(keys=mask_key),
            ScaleIntensityd(keys=keys),
            random_crop,
            RandRotate90d(keys=keys, prob=rotation_prob),
            ToTensord(keys=keys)
        ]),
        valid=Compose([
            LoadImaged(reader=PILReader(), keys=keys),
            AsChannelFirstd(keys=image_key),
            AddChanneld(keys=mask_key),
            ScaleIntensityd(keys=keys),
            ToTensord(keys=keys)
        ]),
        post=Compose([
            Activations(sigmoid=True),
            AsDiscrete(threshold_values=True)
        ])
    )
