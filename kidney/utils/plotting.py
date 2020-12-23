from typing import Optional, Union, Tuple, Dict, cast

import numpy as np
from matplotlib.axis import Axis
from zeus.plotting.utils import axes

from kidney.utils.image import read_masked_image, overlay
from kidney.utils.mask import rle_decode


def plot_mask(
    mask: Union[str, np.ndarray],
    enumerate_pixels: bool = False,
    shape: Optional[Tuple[int, int]] = None,
    ax: Optional[Axis] = None
) -> Union[Axis, np.ndarray]:
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
) -> Axis:
    ax = axes(ax=ax)
    image = read_masked_image(image, mask, **(overlay_config or {}))
    ax.imshow(image)
    ax.axis('off')
    ax.set_title(title)
    return cast(Axis, ax)


def preview_arrays(
    image: np.ndarray,
    gt: Optional[np.ndarray] = None,
    pred: Optional[np.ndarray] = None,
    ax: Optional[Axis] = None,
    overlay_config: Optional[Dict] = None,
    title: str = '',
):
    ax = axes(ax=ax)
    if image.shape[0] in (1, 3):
        image = image.transpose((1, 2, 0))
    overlay_config = (overlay_config or {}).copy()
    if gt is not None:
        overlay_config["color"] = (255, 0, 0)
        image = overlay(image, gt, **overlay_config)
    if pred is not None:
        overlay_config["color"] = (0, 255, 0)
        image = overlay(image, pred, **overlay_config)
    ax.imshow(image)
    ax.axis('off')
    ax.set_title(title)
    return cast(Axis, ax)
