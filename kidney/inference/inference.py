from collections import Callable
from dataclasses import dataclass
from typing import Optional, Generator, Tuple, Dict, Any

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

DEFAULT_WINDOW_SIZE = 1024
DEFAULT_OVERLAP = 64


@dataclass
class SlidingWindowPredictor:
    model: Callable
    transform: Optional[Callable] = None
    post_model: Optional[Callable] = None
    sliding_window_config: Optional[Dict] = None

    def __post_init__(self):
        self.sliding_window_config = self.sliding_window_config or {}

    def predict_from_file(self, filename: str):
        dataset = rasterio.open(filename)
        boxes = sliding_window_boxes(dataset.shape, **self.sliding_window_config)
        prediction = np.zeros(dataset.shape, dtype=np.uint8)
        for x1, y1, x2, y2 in boxes:
            image = dataset.read([1, 2, 3], window=Window.from_slices((x1, x2), (y1, y2)))
            image = np.moveaxis(image, 0, -1)
            image = self._transform_image(image)
            with torch.no_grad():
                prediction[y1:y2, x1:x2] = self._get_scores_from_model(self.model(image))
        return prediction

    def _transform_image(self, image: np.ndarray) -> torch.Tensor:
        return image if self.transform is None else self.transform(image)

    def _get_scores_from_model(self, output: Any) -> torch.Tensor:
        return output if self.post_model is None else self.post_model(output)


def sliding_window_boxes(
    input_shape: Tuple[int, int],
    window_size: int = DEFAULT_WINDOW_SIZE,
    overlap: int = DEFAULT_OVERLAP
) -> np.ndarray:
    h, w = input_shape
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


def main():
    image = np.zeros((3048, 2245))
    for x1, y1, x2, y2 in sliding_window_boxes(image.shape):
        image[y1:y2, x1:x2] = 1
    print(image)


if __name__ == '__main__':
    main()
