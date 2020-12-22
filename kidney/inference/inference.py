import abc
import itertools
from collections import Callable
from dataclasses import dataclass
from enum import auto, Enum
from typing import Optional, Tuple, Dict, Any, Union, List

import cv2 as cv
import numpy as np
import pytorch_lightning as pl
import rasterio
import torch
from rasterio.windows import Window
from zeus.torch_tools.utils import to_np

from kidney.datasets.kaggle import outlier, get_reader
from kidney.parameters import get_relevant_params
from kidney.utils.image import scale_intensity_tensor, channels_first, channels_last


class InferenceAlgorithm:

    @abc.abstractmethod
    def predict_from_file(self, filename: str) -> np.ndarray:
        pass


@dataclass
class SlidingWindowConfig:
    window_size: int
    overlap: int
    max_batch_size: int
    check_for_outliers: bool = True
    transform_input: Optional[Callable] = None
    transform_output: Optional[Callable] = None


@dataclass
class SlidingWindow(InferenceAlgorithm):
    model: pl.LightningModule
    config: SlidingWindowConfig
    device: torch.device

    def predict_from_file(self, filename: str) -> np.ndarray:
        size = self.config.window_size

        with rasterio.open(filename) as dataset:
            w, h = dataset.shape
            boxes = sliding_window_boxes(w, h, size, self.config.overlap)
            n_splits = int(np.ceil(boxes.shape[0] / self.config.max_batch_size))
            mask = np.zeros((h, w), dtype=np.uint8)

            for batch in np.array_split(boxes, n_splits):
                samples = [
                    {
                        "img": self.transform_input(
                            channels_first(
                                cv.resize(
                                    channels_last(image).astype(np.float32),
                                    (size, size)
                                )
                            )
                        ),
                        "box": box
                    }
                    for box, image in (
                        (
                            [x1, y1, x2, y2],
                            dataset.read(
                                [1, 2, 3],
                                window=Window.from_slices((x1, x2), (y1, y2))
                            )
                        )
                        for x1, y1, x2, y2 in batch
                    )
                    if (
                        not self.config.check_for_outliers or
                        self.config.check_for_outliers and not outlier(image)
                    )
                ]
                if not samples:
                    continue
                collated = self.collate_batch(samples)
                output = self.model(collated)
                result = self.transform_output(output)
                self.update_predictions_mask(mask, result, collated["meta"])

        return mask

    def transform_input(self, t: Union[np.ndarray, torch.Tensor]) -> Dict:
        return t if self.config.transform_input is None else self.config.transform_input(t)

    def transform_output(self, x: Dict) -> Dict:
        return x if self.config.transform_output is None else self.config.transform_output(x)

    @classmethod
    def collate_batch(cls, samples: List[Dict]) -> Dict:
        img = torch.stack([sample["img"] for sample in samples])
        meta = [{k: v for k, v in sample.items() if k != "img"} for sample in samples]
        return {"img": img, "meta": meta}

    @classmethod
    def update_predictions_mask(cls, mask: np.ndarray, result: Dict, meta: List[Dict]):
        for i, info in enumerate(meta):
            x1, y1, x2, y2 = info["box"]
            mask[y1:y2, x1:x2] = to_np(result["outputs"][i].byte())


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


def main():
    image = np.zeros((3048, 2245))
    for x1, y1, x2, y2 in sliding_window_boxes(image.shape):
        image[y1:y2, x1:x2] = 1
    print(image)


if __name__ == '__main__':
    main()
