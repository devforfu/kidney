from typing import Optional, Union, Tuple

import cv2 as cv
import numpy as np
from matplotlib.axis import Axis
from zeus.plotting.utils import axes

from kidney.utils.mask import rle_decode


def plot_mask(
    mask: Union[str, np.ndarray],
    enumerate_pixels: bool = False,
    shape: Optional[Tuple[int, int]] = None,
    ax: Optional[Axis] = None
) -> Axis:
    """Plots binary segmentation mask."""

    if isinstance(mask, str):
        assert shape is not None, 'cannot decode RLE without shape provided'
        mask = rle_decode(mask, shape)

    ax = axes(ax)
    ax.imshow(mask, cmap='gray')
    ax.axis('off')

    if enumerate_pixels:
        n, m = shape

        for x in range(m):
            for y in range(n):
                index = x*n + y + 1

                color = 'black' if mask[y][x] == 1 else 'white'

                ax.annotate(str(index), xy=(x, y),
                            ha='center', va='center',
                            color=color, fontsize=18)

    return ax


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
