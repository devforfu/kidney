from dataclasses import dataclass
from typing import Union, Tuple

from monai.data import PILReader
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    ScaleIntensityd,
    RandCropByPosNegLabeld, RandRotate90d, ToTensord, Activations, AsDiscrete
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
    crop_fraction: float = 0.75,
    num_samples: int = 4,
    rotation_prob: float = 0.5
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
    return Transformers(
        train=Compose([
            LoadImaged(reader=PILReader(), keys=keys),
            AddChanneld(keys=keys),
            ScaleIntensityd(keys=image_key),
            RandCropByPosNegLabeld(
                keys=keys,
                label_key=mask_key,
                spatial_size=spatial_size,
                pos=1,
                neg=1,
                num_samples=num_samples
            ),
            RandRotate90d(
                keys=keys,
                prob=rotation_prob,
                spatial_axes=(0, 1)
            ),
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
