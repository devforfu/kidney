import abc
from dataclasses import dataclass
from enum import auto, Enum
from typing import Union, Tuple, Callable, Dict, Mapping, Hashable

import albumentations as A
import torch
import numpy as np
from albumentations.pytorch import ToTensorV2
from monai.data import PILReader
from monai.transforms import (
    Compose, RandCropByPosNegLabeld, RandSpatialCropSamplesd,
    LoadImaged, ScaleIntensityd, RandRotate90d, ToTensord,
    Activations, AsDiscrete, AsChannelFirstd, AddChanneld,
    NormalizeIntensityd, MapTransform, KeysCollection
)
from zeus.core import AutoName

from kidney.utils.image import scale_intensity_tensor, channels_last


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
        output.update({"outputs": transformed.squeeze()})
        return output


@dataclass
class Transformers:
    train: Callable
    valid: Callable
    test_preprocessing: Callable
    test_postprocessing: Callable


class IntensityNormalization(AutoName):
    NoOp = auto()
    ZeroOne = auto()
    TorchvisionSegmentation = auto()

    @classmethod
    def parse(cls, value: str):
        return IntensityNormalization(value)

    def get_stats(self):
        if self == IntensityNormalization.TorchvisionSegmentation:
            return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        return None, None


def create_monai_crop_to_many_sigmoid_transformers(
    image_key: str,
    mask_key: str,
    image_size: Union[int, Tuple[int, int]],
    crop_fraction: float = 0.5,
    num_samples: int = 4,
    rotation_prob: float = 0.5,
    crop_balanced: bool = True,
    pos_fraction: float = 1.0,
    neg_fraction: float = 1.0,
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
    pos_fraction
        Fraction of pixels with segmentation mask.
    neg_fraction
        Fraction of pixels without masked object.
    load_from_disk
        If True, the first step in pipeline reads training sample from disk.
    as_channels_first
        If True, switch the first and the last image channels.
    normalization
        Image normalization method.

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
            pos=pos_fraction, neg=neg_fraction,
            num_samples=num_samples
        )
        if crop_balanced
        else
        RandSpatialCropSamplesd(
            keys=keys, roi_size=spatial_size,
            num_samples=num_samples, random_size=False
        )
    )

    mean, std = normalization.get_stats()
    intensity_normalization = (
        [ScaleIntensityd(keys=image_key)]
        if normalization == IntensityNormalization.ZeroOne else
        [
            ScaleIntensityd(keys=image_key),
            NormalizeGlobalMeanStd(
                keys=image_key,
                mean=np.array(list(mean)).reshape((3, 1, 1)).astype(np.float32),
                std=np.array(list(std)).reshape((3, 1, 1)).astype(np.float32)
            )
        ]
        if normalization == IntensityNormalization.TorchvisionSegmentation else
        []
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


def create_weak_augmentation_transformers(
    image_key: str,
    mask_key: str,
    image_size: int,
    normalization: IntensityNormalization = IntensityNormalization.TorchvisionSegmentation,
    debug: bool = False
) -> Transformers:

    intensity_normalization = create_normalization(normalization, skip=debug)

    final_step = (
        [A.NoOp()]
        if debug
        else [
            intensity_normalization,
            A.Lambda(name="add_channel_axis", mask=add_first_channel),
            ToTensorV2()
        ]
    )

    train_steps = [
        A.Lambda(name="channels_last", image=as_channels_last),
        A.RandomSizedCrop(
            min_max_height=(image_size*3//4, image_size),
            height=image_size,
            width=image_size
        ),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.3),
        *final_step
    ]

    valid_steps = [
        A.Lambda(name="channels_last", image=as_channels_last),
        A.PadIfNeeded(min_height=image_size, min_width=image_size),
        *final_step
    ]

    return Transformers(
        train=AlbuAdapter(A.Compose(train_steps), image_key, mask_key),
        valid=AlbuAdapter(A.Compose(valid_steps), image_key, mask_key),
        test_preprocessing=AlbuAdapter(A.Compose(valid_steps), image_key, mask_key),
        test_postprocessing=SigmoidOutputAsMask(),
    )


def create_strong_augmentation_transformers(
    image_key: str,
    mask_key: str,
    image_size: int,
    normalization: IntensityNormalization = IntensityNormalization.TorchvisionSegmentation,
    debug: bool = False
):
    intensity_normalization = create_normalization(normalization, skip=debug)

    final_step = (
        [A.NoOp()]
        if debug
        else [
            intensity_normalization,
            A.Lambda(name="add_channel_axis", mask=add_first_channel),
            ToTensorV2()
        ]
    )

    train_steps = [
        A.Lambda(name="channels_last", image=as_channels_last),
        A.Resize(image_size, image_size),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),
        A.ColorJitter(
            brightness=0.07,
            contrast=0.07,
            saturation=0.1,
            hue=0.1,
            p=0.3
        ),
        *final_step
    ]

    valid_steps = [
        A.Lambda(name="channels_last", image=as_channels_last),
        A.Resize(image_size, image_size),
        *final_step
    ]

    return Transformers(
        train=AlbuAdapter(A.Compose(train_steps), image_key, mask_key),
        valid=AlbuAdapter(A.Compose(valid_steps), image_key, mask_key),
        test_preprocessing=AlbuAdapter(A.Compose(valid_steps), image_key, mask_key),
        test_postprocessing=SigmoidOutputAsMask()
    )


def create_normalization(normalization: IntensityNormalization, skip: bool = False):
    if skip:
        return A.NoOp()
    mean, std = [
        np.array(list(stat)).reshape((1, 1, 3))
        for stat in normalization.get_stats()
    ]
    return A.Normalize(mean, std)


def add_first_channel(arr: np.ndarray, **kwargs) -> np.ndarray:
    return arr[np.newaxis]


def as_channels_last(arr: np.ndarray, **kwargs) -> np.ndarray:
    return channels_last(arr)


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


@dataclass
class AlbuAdapter:
    callable: Callable
    image_key: str
    mask_key: str

    def __call__(self, inputs: Dict) -> Dict:
        output = self.callable(
            image=inputs[self.image_key],
            mask=inputs[self.mask_key]
        )
        adapted = {
            self.image_key: output["image"],
            self.mask_key: output["mask"]
        }
        return adapted
