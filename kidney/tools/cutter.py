"""Cuts large segmentation images into smaller pieces."""
import os
from typing import Optional, Dict

import cv2 as cv
import numpy as np
import PIL.Image

from kidney.datasets.kaggle import outlier, DatasetReader, SampleType


class OutliersFilter:
    """Excludes images that don't include relevant information."""

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> bool:
        raise NotImplementedError()


class SaturationFilter(OutliersFilter):

    def __init__(self, max_saturation: float = 40, min_pixels: int = 1000):
        self.max_saturation = max_saturation
        self.min_pixels = min_pixels

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> bool:
        hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
        _, saturation, _ = cv.split(hsv)

        if (saturation > self.max_saturation).sum() <= self.min_pixels:
            return True

        if image.sum() <= self.min_pixels:
            return True

        return False


class HistogramFilter(OutliersFilter):

    def __init__(self, bin_size: int = 4, threshold: int = 1000):
        self.bin_size = bin_size
        self.threshold = threshold

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> bool:
        return outlier(image, self.bin_size, self.threshold)


class NoOpFilter(OutliersFilter):

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> bool:
        return False


def pad_sample(sample: Dict, tile_size: int, reduce: int = 1) -> Dict:
    img, seg = sample["img"], sample.get("seg")

    h, w = img.shape[:2]
    sz = tile_size
    mod = reduce * sz
    pad_y = (mod - h % mod) % mod
    pad_x = (mod - w % mod) % mod
    padding = [[pad_y//2, pad_y - pad_y//2], [pad_x//2, pad_x - pad_x//2]]

    img = np.pad(img, padding + [[0, 0]], constant_values=0)
    if seg is not None:
        seg = np.pad(seg, padding, constant_values=0)

    new_size = img.shape[1]//reduce, img.shape[0]//reduce
    img = cv.resize(img, new_size, interpolation=cv.INTER_AREA)
    img = img.reshape(img.shape[0]//sz, sz, img.shape[1]//sz, sz, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)
    if seg is not None:
        seg = cv.resize(seg, new_size, interpolation=cv.INTER_NEAREST)
        seg = seg.reshape(seg.shape[0]//sz, sz, seg.shape[1]//sz, sz)
        seg = seg.transpose(0, 2, 1, 3).reshape(-1, sz, sz)

    return {"img": img, "seg": seg}


def cut_sample(
    sample: Dict,
    tile_size: int,
    outliers_filter: OutliersFilter = NoOpFilter(),
    reduce: int = 1,
):
    padded = pad_sample(
        sample={"img": sample["image"], "seg": sample["mask"]},
        tile_size=tile_size,
        reduce=reduce
    )
    crops = []
    for i, (image, mask) in enumerate(zip(padded["img"], padded["seg"])):
        if outliers_filter(image, mask):
            continue
        crop = dict(img=PIL.Image.fromarray(image))
        if mask is not None:
            crop["seg"] = PIL.Image.fromarray(mask)
        crops.append(crop)
    return crops
