from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import rasterio


def sliding_window_boxes(
    width: int,
    height: int,
    window_size: int,
    overlap: int
) -> np.ndarray:
    w, h = width, height
    x_size = w // (window_size - overlap) + 1
    y_size = h // (window_size - overlap) + 1

    x1 = np.linspace(0, w, num=x_size, endpoint=False, dtype=np.int32)
    x1[-1] = w - window_size
    x2 = (x1 + window_size).clip(0, w)

    y1 = np.linspace(0, h, num=y_size, endpoint=False, dtype=np.int32)
    y1[-1] = h - window_size
    y2 = (y1 + window_size).clip(0, h)

    boxes = np.column_stack([
        np.dstack(np.meshgrid(x1, y1)).reshape(-1, 2),
        np.dstack(np.meshgrid(x2, y2)).reshape(-1, 2)
    ])

    return boxes


@dataclass
class SlidingWindowsGenerator:
    window_size: int
    overlap: int
    limit: Optional[int] = None

    def generate(self, filename: str) -> Tuple[np.ndarray, Tuple[int, int]]:
        identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
        with rasterio.open(filename, transform=identity) as dataset:
            height, width = shape = dataset.shape
        boxes = sliding_window_boxes(width, height, self.window_size, self.overlap)
        if self.limit is not None:
            subset = np.random.choice(np.arange(boxes.shape[0]), self.limit, replace=False)
            boxes = boxes[subset]
        return boxes, shape
