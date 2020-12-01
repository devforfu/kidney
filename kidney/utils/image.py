from typing import Tuple, Optional, Dict

import cv2 as cv
import numpy as np
import PIL.Image


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
    color = np.asarray(color).reshape(1, 1, 3)
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
