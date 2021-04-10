import abc
from dataclasses import dataclass
from enum import auto
from typing import Union, Tuple, Callable, Dict, Mapping, Hashable, Optional

import albumentations as A
import copy
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from monai.data import PILReader
from monai.transforms import (
    Compose, RandCropByPosNegLabeld, RandSpatialCropSamplesd,
    LoadImaged, ScaleIntensityd, RandRotate90d, ToTensord,
    Activations, AsDiscrete, AsChannelFirstd, AddChanneld,
    MapTransform, KeysCollection
)
from pytorch_lightning.utilities import AttributeDict
from zeus.core import AutoName
from zeus.utils import if_none

from kidney.datasets.color_transfer import ColorTransferAugmentation
from kidney.parameters import requires
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


def create_color_augmentation_transformers(
    image_key: str,
    mask_key: str,
    image_size: int,
    normalization: IntensityNormalization = IntensityNormalization.TorchvisionSegmentation,
    color_transfer: Optional[str] = None,
    debug: bool = False,
) -> Transformers:

    intensity_normalization = create_normalization(normalization, skip=debug)

    if color_transfer is not None:
        color_transfer = ColorTransferAugmentation(color_transfer)

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
        A.Lambda(name="color_transfer", image=color_transfer) if color_transfer else A.NoOp(),
        A.Resize(image_size, image_size),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=45, p=0.25),
        A.ElasticTransform(p=0.5),
        A.ToGray(p=0.25),
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


@dataclass
class AugConfig:
    prob: float = 1.0


@dataclass
class BCG(AugConfig):
    brightness: float = 0.2
    contrast: float = 0.2
    gamma: Tuple[float, float] = (80, 120)


@dataclass
class Blur(AugConfig):
    limit: int = 3


@dataclass
class Noise(AugConfig):
    gauss_var: float = 0.001
    affine_scale: Optional[float] = None


@dataclass
class Flip(AugConfig):
    vertical: bool = True
    horizontal: bool = True


@dataclass
class ShiftScaleRotate(AugConfig):
    shift: Optional[float] = 0.0625
    scale: Optional[float] = 0.1
    rotate: Optional[int] = 45


@dataclass
class Elastic(AugConfig):
    alpha: int = 1
    sigma: int = 50
    alpha_affine: int = 50


def create_transformers_v2(
    image_size: Optional[int] = None,
    bcg: Optional[BCG] = BCG(),
    blur: Optional[Blur] = Blur(),
    noise: Optional[Noise] = Noise(),
    flip: Optional[Flip] = Flip(),
    elastic: Optional[Elastic] = Elastic(),
    shift_scale_rotate: Optional[ShiftScaleRotate] = ShiftScaleRotate(),
    color_transfer: Optional[ColorTransferAugmentation] = None,
    normalization: IntensityNormalization = IntensityNormalization.TorchvisionSegmentation,
    debug: bool = False,
) -> Transformers:

    train_steps, valid_steps = [], []

    if image_size is not None:
        train_steps.append(A.Resize(image_size, image_size))
        valid_steps.append(A.Resize(image_size, image_size))

    if bcg is not None:
        train_steps.append(A.OneOf([
            A.RandomBrightnessContrast(bcg.brightness, bcg.contrast, p=1),
            A.RandomGamma(bcg.gamma, p=1)
        ], p=bcg.prob))

    if blur is not None:
        train_steps.append(A.OneOf([
            A.Blur(blur_limit=blur.limit),
            A.MedianBlur(blur_limit=blur.limit)
        ], p=blur.prob))

    if noise is not None:
        train_steps.append(A.OneOf(
            [A.GaussNoise(var_limit=noise.gauss_var)]
            if noise.affine_scale is None
            else [
                A.GaussNoise(var_limit=noise.gauss_var),
                A.IAAAffine(scale=noise.affine_scale)
            ],
            p=noise.prob
        ))

    if flip is not None:
        flips = []
        if flip.vertical:
            flips.append(A.VerticalFlip(p=flip.prob))
        if flip.horizontal:
            flips.append(A.HorizontalFlip(p=flip.prob))
        train_steps.extend(flips)

    if elastic is not None:
        train_steps.append(A.ElasticTransform(
            alpha=elastic.alpha,
            sigma=elastic.sigma,
            alpha_affine=elastic.alpha_affine,
            p=elastic.prob
        ))

    if shift_scale_rotate is not None:
        train_steps.append(A.ShiftScaleRotate(
            shift_limit=shift_scale_rotate.shift,
            scale_limit=shift_scale_rotate.scale,
            rotate_limit=shift_scale_rotate.rotate,
            p=shift_scale_rotate.prob
        ))

    if color_transfer is not None:
        train_steps.append(A.Lambda(
            name="color_transfer",
            image=color_transfer,
            p=color_transfer.prob
        ))

    final_step = (
        [A.NoOp()]
        if debug
        else [
            create_normalization(normalization, skip=debug),
            A.Lambda(name="add_channel_axis", mask=add_first_channel),
            ToTensorV2()
        ]
    )

    train_steps.extend(final_step)
    valid_steps.extend(copy.deepcopy(final_step))

    return Transformers(
        train=A.Compose(train_steps),
        valid=A.Compose(valid_steps),
        test_preprocessing=A.Compose(valid_steps),
        test_postprocessing=SigmoidOutputAsMask()
    )


def create_color_strong_augmentation_transformers(
    image_key: str,
    mask_key: str,
    image_size: int,
    normalization: IntensityNormalization = IntensityNormalization.TorchvisionSegmentation,
    color_transfer: Optional[str] = None,
    debug: bool = False,
) -> Transformers:
    intensity_normalization = create_normalization(normalization, skip=debug)

    if color_transfer is not None:
        color_transfer = ColorTransferAugmentation(color_transfer)

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
        # A.Lambda(name="channels_last", image=as_channels_last),
        # A.Lambda(name="color_transfer", image=color_transfer) if color_transfer else A.NoOp(),
        A.Resize(image_size, image_size),
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1)
        ], p=0.5),
        A.OneOf([
            A.Blur(blur_limit=3, p=1),
            A.MedianBlur(blur_limit=3, p=1)
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=0.002, p=0.5),
            A.IAAAffine(p=0.5)
        ], p=0.25),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=45, p=0.25),
        A.ElasticTransform(p=0.25),
        A.ShiftScaleRotate(p=0.25),
        A.ToGray(p=0.25),
        *final_step
    ]

    valid_steps = [
        # A.Lambda(name="channels_last", image=as_channels_last),
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
    if skip or normalization == IntensityNormalization.NoOp:
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


@requires([
    "model_input_size",
    "model_input_image_key",
    "model_input_mask_key",
    "aug_pipeline",
    "aug_normalization_method"
])
def get_transformers(params: AttributeDict) -> Transformers:
    from kidney.datasets.utils import get_dataset_input_size

    if params.aug_pipeline == "v2":
        return create_transformers_v2(
            image_size=if_none(
                params.model_input_size,
                get_dataset_input_size(params.dataset)
            ),
            normalization=params.aug_normalization_method
        )
    try:
        return {
            "weak": create_weak_augmentation_transformers,
            "strong": create_strong_augmentation_transformers,
            "color": create_color_augmentation_transformers,
            "color_strong": create_color_strong_augmentation_transformers,
        }[params.aug_pipeline](
            image_key=params.model_input_image_key,
            mask_key=params.model_input_mask_key,
            image_size=if_none(
                params.model_input_size,
                get_dataset_input_size(params.dataset)
            ),
            normalization=params.aug_normalization_method
        )
    except KeyError:
        raise ValueError(f"unknown pipeline name: {params.aug_pipeline}")
