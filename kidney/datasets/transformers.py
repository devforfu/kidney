import abc
from dataclasses import dataclass
from enum import auto, Enum
from typing import Union, Tuple, Callable, Dict, Mapping, Hashable

import torch
import numpy as np
from monai.data import PILReader
from monai.transforms import (
    Compose, RandCropByPosNegLabeld, RandSpatialCropSamplesd,
    LoadImaged, ScaleIntensityd, RandRotate90d, ToTensord,
    Activations, AsDiscrete, AsChannelFirstd, AddChanneld,
    NormalizeIntensityd, MapTransform, KeysCollection
)

from kidney.utils.image import scale_intensity_tensor


class InputPreprocessor:

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.prepare(tensor)

    @abc.abstractmethod
    def prepare(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class OutputPostprocessor:

    def __call__(self, output: Dict) -> Dict:
        return self.prepare(output)

    @abc.abstractmethod
    def prepare(self, output: Dict) -> Dict:
        raise NotImplementedError()


class ImageNormalization(InputPreprocessor):
    """Converts input image into 0-1 normalized tensor."""

    def prepare(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(tensor)
        tensor = tensor.float()
        tensor = scale_intensity_tensor(tensor)
        return tensor


class SigmoidOutputAsMask(OutputPostprocessor):

    def __init__(self):
        self.as_discrete = Compose([
            Activations(sigmoid=True),
            AsDiscrete(threshold_values=True)
        ])

    def prepare(self, output: Dict) -> Dict:
        transformed = self.as_discrete(output["outputs"])
        output.update({"outputs": transformed.squeezed()})
        return output


@dataclass
class Transformers:
    train: Callable
    valid: Callable
    test_preprocessing: Callable
    test_postprocessing: Callable


class IntensityNormalization(Enum):
    ZeroOne = auto()
    ImageNet = auto()


def create_monai_crop_to_many_sigmoid_transformers(
    image_key: str,
    mask_key: str,
    image_size: Union[int, Tuple[int, int]],
    crop_fraction: float = 0.5,
    num_samples: int = 4,
    rotation_prob: float = 0.5,
    crop_balanced: bool = True,
    load_from_disk: bool = True,
    as_channels_first: bool = True,
    normalization: IntensityNormalization = IntensityNormalization.ZeroOne,
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
    load_from_disk
        If True, the first step in pipeline reads training sample from disk.

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

    intensity_normalization = (
        [ScaleIntensityd(keys=image_key)]
        if normalization == IntensityNormalization.ZeroOne
        else
        [
            ScaleIntensityd(keys=image_key),
            NormalizeGlobalMeanStd(
                keys=image_key,
                mean=np.array([0.625, 0.448, 0.688]).reshape((3, 1, 1)).astype(np.float32),
                std=np.array([0.131, 0.177, 0.101]).reshape((3, 1, 1)).astype(np.float32)
            )
        ]
    )

    train_steps = [
        AddChanneld(keys=mask_key),
        *intensity_normalization,
        random_crop,
        RandRotate90d(keys=keys, prob=rotation_prob),
        ToTensord(keys=keys)
    ]

    valid_steps = [
        AddChanneld(keys=mask_key),
        *intensity_normalization,
        ToTensord(keys=keys)
    ]

    if as_channels_first:
        train_steps.insert(0, AsChannelFirstd(keys=image_key))
        valid_steps.insert(0, AsChannelFirstd(keys=image_key))

    if load_from_disk:
        train_steps.insert(0, LoadImaged(reader=PILReader(), keys=keys))
        valid_steps.insert(0, LoadImaged(reader=PILReader(), keys=keys))

    return Transformers(
        train=Compose(train_steps),
        valid=Compose(valid_steps),
        test_preprocessing=Compose(intensity_normalization),
        test_postprocessing=SigmoidOutputAsMask()
    )


class NormalizeGlobalMeanStd(MapTransform):

    def __init__(
        self,
        keys: KeysCollection,
        mean: np.ndarray,
        std: np.ndarray
    ):
        super().__init__(keys)
        self.mean = mean
        self.std = std

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.normalize(d[key])
        return d

    def normalize(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.mean)/self.std
