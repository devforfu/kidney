import abc
from collections import Callable
from dataclasses import dataclass
from typing import Optional, Dict, Union, List

import cv2 as cv
import numpy as np
import pytorch_lightning as pl
import rasterio
import torch
from rasterio.windows import Window
from zeus.torch_tools.utils import to_np

from kidney.datasets.kaggle import outlier, DatasetReader, SampleType
from kidney.inference.window import sliding_window_boxes
from kidney.utils.image import channels_first, channels_last


class InferenceAlgorithm:

    @abc.abstractmethod
    def predict_from_file(self, filename: str) -> np.ndarray:
        pass

    def predict_from_reader(
        self,
        reader: DatasetReader,
        sample_type: SampleType,
        sample_path_key: str = "tiff",
        encoder: Optional[Callable] = None
    ) -> List[Dict]:
        """Runs predictions dataset.

        Parameters
        ----------
        reader
            Extracts samples from storage.
        sample_type
            Predict on selected sample type only.
        sample_path_key
            Sample meta-information key with path to the sample.
        encoder
            If provided, a function applied to the generated prediction.

        Returns
        -------
        predictions
            The list of dictionaries with predictions and their IDs.

        """
        predictions = []
        for key in reader.get_keys(sample_type):
            meta = reader.fetch_meta(key)
            predicted = self.predict_from_file(meta[sample_path_key])
            if encoder is not None:
                predicted = encoder(predicted)
            predictions.append({"id": key, "predicted": predicted})
        return predictions


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
