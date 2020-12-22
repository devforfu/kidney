import abc
from collections import Callable
from dataclasses import dataclass
from enum import auto, Enum
from typing import Optional, Tuple, Dict, Any, Union

import cv2 as cv
import numpy as np
import pytorch_lightning as pl
import rasterio
import torch
from rasterio.windows import Window
from zeus.torch_tools.utils import to_np

from kidney.datasets.kaggle import outlier, get_reader
from kidney.parameters import get_relevant_params
from kidney.utils.image import scale_intensity_tensor, channels_first

DEFAULT_WINDOW_SIZE = 1024
DEFAULT_OVERLAP = 64


class InferenceAlgorithm(Enum):
    Default = auto()


class InferenceProtocol:

    @abc.abstractmethod
    def predict_from_file(self, filename: str) -> np.ndarray:
        pass


def create_inference_instance(
    algorithm: InferenceAlgorithm,
    experiment: pl.LightningModule,
    **config
) -> InferenceProtocol:
    config = config or {}

    if algorithm == InferenceAlgorithm.Default:
        device = torch.device(config.get("device", "cpu"))
        experiment = experiment.to(device)
        transformers = config.get("transformers")

        def transform(tensor: Union[np.ndarray, torch.Tensor]) -> Dict:
            nonlocal device
            tensor = torch.as_tensor(tensor)
            tensor = scale_intensity_tensor(tensor)
            tensor = tensor.to(device)
            return {"img": tensor}

        def postprocessing(model_output: Dict) -> np.ndarray:
            nonlocal transformers
            outputs = model_output["outputs"]
            if transformers is not None:
                outputs = transformers.post(outputs)
            return to_np(outputs.squeeze())

        predictor = SlidingWindowPredictor(
            model=experiment,
            transform=transform,
            post_model=postprocessing,
            **get_relevant_params(SlidingWindowPredictor, config)
        )

        return predictor

    raise NotImplementedError(f"unknown algorithm: {algorithm}")







@dataclass
class SlidingWindowPredictor(InferenceProtocol):
    model: Callable
    transform: Optional[Callable] = None
    post_model: Optional[Callable] = None
    check_for_outliers: bool = False
    window_size: int = DEFAULT_WINDOW_SIZE
    overlap: int = DEFAULT_OVERLAP,
    max_batch_size: int = 16

    def __post_init__(self):
        self.sliding_window_config = self.sliding_window_config or {}

    def predict_from_file(self, filename: str) -> np.ndarray:
        with rasterio.open(filename) as dataset:
            w, h = dataset.shape
            boxes = sliding_window_boxes(w, h, self.window_size, self.overlap)
            n_splits = int(np.ceil(boxes.shape[0] / self.max_batch_size))
            prediction = np.zeros((h, w), dtype=np.uint8)

            for batch in np.array_split(boxes, n_splits):
                images = [
                    self.transform(image)
                    for image in (
                        dataset.read(
                            [1, 2, 3],
                            window=Window.from_slices((x1, x2), (y1, y2))
                        )
                        for x1, y1, x2, y2 in batch
                    )
                    if (
                        not self.check_for_outliers or
                        self.check_for_outliers and not outlier(image)
                    )
                ]


        return np.zeros((4, 4))

    # def predict_from_file(self, filename: str):
    #     dataset = rasterio.open(filename)
    #     w, h = dataset.shape
    #     boxes = sliding_window_boxes((h, w), **self.sliding_window_config)
    #     prediction = np.zeros(dataset.shape, dtype=np.uint8)
    #     for x1, y1, x2, y2 in boxes:
    #         window = Window.from_slices((x1, x2), (y1, y2))
    #         image = dataset.read([1, 2, 3], window=window)
    #         shape = (y2 - y1, x2 - x1)
    #         if self.check_for_outliers and outlier(image):
    #             mask = np.zeros(shape, dtype=np.uint8)
    #         else:
    #             image = np.moveaxis(image, 0, -1)
    #             image = cv.resize(image, shape)
    #             image = channels_first(image)
    #             image = image[np.newaxis]
    #             batch = self._transform_image(image)
    #             with torch.no_grad():
    #                 mask = self._get_scores_from_model(self.model(batch))
    #                 print(mask.shape)
    #         prediction[y1:y2, x1:x2] = mask
    #     return prediction

    def _transform_image(self, image: np.ndarray) -> torch.Tensor:
        return image if self.transform is None else self.transform(image)

    def _get_scores_from_model(self, output: Any) -> torch.Tensor:
        return output if self.post_model is None else self.post_model(output)


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
