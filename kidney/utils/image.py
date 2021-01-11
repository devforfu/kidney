import fnmatch
import glob
import os
import random
from typing import Tuple, Optional, Dict, Set

import cv2 as cv
import numpy as np
import pandas as pd
import PIL.Image
import torch


def detect_roi(
    image: np.ndarray,
    thumbnail_size: Tuple[int, int] = (1024, 1024)
) -> Tuple[int, int, int, int]:
    """Detects the biggest contour on the image.

    Helpful to get rid of non-informative boundary around the training images.

    Args:
        image: Training image to cut the boundaries.
        thumbnail_size: Size to reduce the image before running contours detector.

    Returns:
        Tuple: The detected box.

    """
    thumbnail = cv.resize(image, thumbnail_size)
    gray = cv.cvtColor(thumbnail, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_OTSU + cv.THRESH_BINARY)[1]

    contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) else contours[1]
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    biggest = contours[0]

    xr, yr, wr, hr = cv.boundingRect(biggest)
    w0, h0 = image.shape[1], image.shape[0]
    wt, ht = thumbnail_size
    rw, rh = w0/wt, h0/ht
    x0, y0, x1, y1 = [int(v) for v in [
        (xr * rw),
        (yr * rh),
        (xr + wr) * rw,
        (yr + hr) * rh
    ]]

    return x0, y0, x1, y1


def overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
    resize: Tuple[int, int] = (1024, 1024)
) -> np.ndarray:
    """Combines image and its segmentation mask into a single image.

    Params:
        image: Training image.
        mask: Segmentation mask.
        color: Color for segmentation mask rendering.
        alpha: Segmentation mask's transparency.
        resize: If provided, both image and its mask are resized before blending them together.

    Returns:
        image_combined: The combined image.

    """
    assert image.ndim == 3, "image array should have three dimensions"
    if image.shape[-1] == 1:
        image = image.repeat(3, axis=-1)

    color = np.asarray(color).reshape((1, 1, 3))
    colored_mask = np.expand_dims(mask, -1).repeat(3, axis=-1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv.resize(image, resize)
        image_overlay = cv.resize(image_overlay, resize)

    image_combined = cv.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


def read_image_as_numpy(path: str) -> np.ndarray:
    return np.asarray(PIL.Image.open(path))


def read_masked_image(
    image: str,
    mask: Optional[str] = None,
    **overlay_options: Dict
) -> np.ndarray:
    image = read_image_as_numpy(image)
    if mask is not None:
        mask = read_image_as_numpy(mask)
    return overlay(image, mask, **overlay_options)


def pixel_histogram(image: np.ndarray, bin_size: int = 4) -> pd.Series:
    return (
        pd.cut(
            image.ravel(),
            bins=range(0, 256 + bin_size, bin_size),
            labels=range(0, 256, bin_size),
            right=False
        )
        .value_counts()
        .rename('count')
    )


def random_image_shape(folder: str) -> Tuple[int, ...]:
    """Picks a random image from a folder and computes its size.

    In case if all images in the folder are of the same size, this function
    helps quickly get dataset's samples shape. The function doesn't ensure
    that all images do *actually* have the same size though.
    """
    filename = random.choice(glob.glob(f"{folder}/*.png"))
    arr = read_image_as_numpy(filename)
    return arr.shape


def compute_image_sizes(root: str) -> Set[Tuple[int, ...]]:
    """Computes unique image sizes for samples in a folder."""

    files = (os.path.join(root, fn) for fn in os.listdir(root))
    sizes = {PIL.Image.open(fn).size for fn in files if fnmatch.fnmatch(fn, "*.png")}
    return sizes


def channels_first(image: np.ndarray) -> np.ndarray:
    """Transforms image into channels-first format if it isn't yet."""
    if image.ndim == 2:
        return image[np.newaxis, :]
    elif image.ndim == 3:
        if image.shape[0] in (1, 3):
            return image  # already in channels-first format
        if image.shape[-1] in (1, 3):
            return image.transpose((2, 0, 1))
    raise ValueError(f"wrong image shape: {image.shape}")


def channels_last(image: np.ndarray) -> np.ndarray:
    """Transforms image into channels-last format if it isn't yet."""

    if image.ndim == 2:
        return image[:, np.newaxis]
    elif image.ndim == 3:
        if image.shape[0] in (1, 3):
            return image.transpose((1, 2, 0))
    raise ValueError(f"wrong image shape: {image.shape}")


def scale_intensity_tensor(tensor: torch.Tensor, scale_range: Tuple[float, float] = (0.0, 1.0)):
    tensor = tensor.float()
    lo, hi = tensor.min(), tensor.max()
    if lo == hi:
        return tensor.float() * lo
    range_lo, range_hi = scale_range
    tensor.sub_(lo).div_(hi - lo)
    tensor.mul_(range_hi - range_lo).add_(range_lo)
    return tensor


def pil_read_image(path: str) -> np.array:
    return np.asarray(PIL.Image.open(path))
