from typing import Optional, Union, Tuple, Dict

import cv2 as cv
import numpy as np
import PIL.Image
from matplotlib.axis import Axis
from zeus.plotting.utils import axes

from kidney.utils.image import read_masked_image
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


def preview(
    image: str,
    mask: Optional[str] = None,
    ax: Optional[Axis] = None,
    overlay_config: Optional[Dict] = None,
    title: str = '',
):
    ax = axes(ax=ax)
    image = read_masked_image(image, mask, **(overlay_config or {}))
    ax.imshow(image)
    ax.axis('off')
    ax.set_title(title)
    return ax
