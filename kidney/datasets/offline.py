from collections import OrderedDict
from enum import auto
from typing import List, Dict, Callable, Optional, Tuple, Type

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from zeus.core import AutoName

from kidney.datasets.kaggle import DatasetReader, SampleType
from kidney.datasets.transformers import Transformers
from kidney.datasets.utils import create_train_valid_data_loaders
from kidney.utils.image import pil_read_image


class GrayscaleProcessing(AutoName):
    ConvertToRGB = auto()
    AsIs = auto()


class OfflineCroppedDataset(Dataset):

    def __init__(
        self,
        samples: List[Dict],
        transform: Optional[Callable] = None,
        read_image_fn: Callable = pil_read_image,
        grayscale_processing: GrayscaleProcessing = GrayscaleProcessing.ConvertToRGB,
    ):
        super().__init__()
        self.samples = samples
        self.transform = transform
        self.read_image_fn = read_image_fn
        self.grayscale_processing = grayscale_processing

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item: int) -> Dict:
        img, seg = self.read_images(self.samples[item])
        sample = {"img": img, "seg": seg}
        return sample if self.transform is None else self.transform(sample)

    def read_images(self, sample: Dict) -> Dict:
        img = self.read_image_fn(sample["img"])
        if img.ndim == 2 and self.grayscale_processing == GrayscaleProcessing.ConvertToRGB:
            img = np.repeat(img[..., np.newaxis], 3, -1)
        if "seg" in sample:
            seg = self.read_image_fn(sample["seg"])
        else:
            seg = np.zeros(img.shape[:2])
        img, seg = [arr.astype(np.float32) for arr in (img, seg)]
        if seg.max() > 1:
            seg /= 255
        return img, seg


class OfflineCroppedDatasetV2(Dataset):
    """An updated version of the dataset that better suitable to be used with Albumentations.

    The dataset doesn't automatically convert grayscale images into RGB, but prepares segmentation masks as
    the old dataset is doing. Also, it reads images using OpenCV library.
    """
    _default_keys_mapping = {"image": "img", "mask": "seg"}

    def __init__(
        self,
        samples: List[Dict],
        transform: Optional[Callable] = None,
        keys_mapping: Optional[Dict] = None
    ):
        super().__init__()
        self.samples = samples
        self.transform = transform
        self.keys_mapping = keys_mapping or self._default_keys_mapping

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item: int) -> Dict:
        image, mask = self.read_sample(item)
        if self.transform is None:
            return {"img": _float32(image), "seg": _float32(mask)}
        else:
            transformed = self.transform(image=image, mask=mask)
            transformed = {self.keys_mapping.get(k, k): v for k, v in transformed.items()}
            image, mask = transformed["img"], transformed["seg"]
            if torch.is_tensor(image):
                image, mask = image.float(), mask.float()
            else:
                image, mask = _float32(image), _float32(mask)
            return {"img": image, "seg": mask}

    def read_sample(self, item: int) -> Tuple[np.ndarray, np.ndarray]:
        image = self.read_image(item)
        mask = self.read_mask(item)
        if mask is None:
            mask = np.zeros(image.shape[:2])
        return image, mask

    def read_image(self, item: int) -> np.ndarray:
        image = cv.imread(self.samples[item]["img"])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        return image

    def read_mask(self, item: int) -> Optional[np.ndarray]:
        filename = self.samples[item].get("seg")
        if filename is None:
            return None
        mask = cv.imread(filename, cv.IMREAD_UNCHANGED)
        return _ensure_binary_mask(mask)


def _ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 1, 0)


def _float32(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32)


def create_data_loaders(
    reader: DatasetReader,
    transformers: Transformers,
    samples: List[Dict],
    train_keys: Optional[List[str]] = None,
    valid_keys: Optional[List[str]] = None,
    num_workers: int = 0,
    batch_size: int = 4,
    dataset_version: int = 2,
    **extra
) -> OrderedDict:
    assert dataset_version in (1, 2)

    def dataset_factory(name: str, subset_samples: List[Dict]) -> Dataset:
        transform = getattr(transformers, name, None)
        return (OfflineCroppedDataset if dataset_version == 1 else OfflineCroppedDatasetV2)(
            samples=subset_samples,
            transform=transform
        )

    return create_train_valid_data_loaders(
        keys=reader.get_keys(SampleType.Labeled),
        transformers=transformers,
        dataset_factory=dataset_factory,
        samples=samples,
        train_keys=train_keys,
        valid_keys=valid_keys,
        num_workers=num_workers,
        batch_size=batch_size,
        **extra
    )
