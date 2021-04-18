import abc
from collections import Callable
from dataclasses import dataclass
from typing import Optional, Dict, List, cast

import cv2 as cv
import numpy as np
import pytorch_lightning as pl
import rasterio
import torch
from rasterio.windows import Window
from zeus.torch_tools.utils import to_np

from kidney.datasets.kaggle import outlier, DatasetReader, SampleType
from kidney.datasets.transformers import AlbuAdapter, as_channels_last, OutputPostprocessor
from kidney.inference.window import sliding_window_boxes


class InferenceAlgorithm:
    @abc.abstractmethod
    def predict_from_file(self, filename: str) -> np.ndarray:
        pass

    def predict_from_reader(
        self,
        reader: DatasetReader,
        sample_type: SampleType,
        sample_path_key: str = "tiff",
        encoder: Optional[Callable] = None,
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
        keys = reader.get_keys(sample_type)
        for i, key in enumerate(keys):
            print(f"processing key: {key} [{i + 1}/{len(keys)}]")
            meta = reader.fetch_meta(key)
            predicted = self.predict_from_file(meta[sample_path_key])
            if encoder is not None:
                predicted = encoder(predicted)
            predictions.append({"id": key, "predicted": predicted})
        return predictions


@dataclass
class SlidingWindowConfig:
    window_size: int = 1024
    overlap: int = 32
    max_batch_size: int = 32
    check_for_outliers: bool = True
    outliers_threshold: int = 1000
    transform_input: Optional[Callable] = None
    transform_output: Optional[Callable] = None


@dataclass
class SlidingWindow(InferenceAlgorithm):
    model: pl.LightningModule
    config: SlidingWindowConfig
    device: torch.device = torch.device("cpu")
    debug: bool = False

    def __post_init__(self):
        self.model = cast(pl.LightningModule, self.model.to(self.device))

    def predict_from_file(self, filename: str) -> np.ndarray:
        size = self.config.window_size

        with rasterio.open(filename) as dataset:
            h, w = dataset.shape
            boxes = sliding_window_boxes(w, h, size, self.config.overlap)
            n_splits = int(np.ceil(boxes.shape[0] / self.config.max_batch_size))
            mask = np.zeros((h, w), dtype=np.uint8)

            print("boxes widths:", set(boxes[:, 2] - boxes[:, 0]))
            print("boxes heights:", set(boxes[:, 3] - boxes[:, 1]))

            for i, batch in enumerate(np.array_split(boxes, n_splits), 1):
                print(f"processing batch {i} of {n_splits}... ", end="")
                samples = [
                    {
                        "box": box,
                        **self.transform_input(
                            {"img": to_rgb(image), "seg": np.zeros((size, size))}
                        ),
                    }
                    for box, image in (
                        (
                            [x1, y1, x2, y2],
                            dataset.read(
                                dataset.indexes,
                                window=Window.from_slices((y1, y2), (x1, x2)),
                            ),
                        )
                        for x1, y1, x2, y2 in batch
                    )
                    if (
                        not self.config.check_for_outliers
                        or (
                            self.config.check_for_outliers
                            and not outlier(
                                image, threshold=self.config.outliers_threshold
                            )
                        )
                    )
                ]
                if samples:
                    print(f"number of samples: {len(samples)}")
                else:
                    print("no relevant samples")
                    continue

                with torch.no_grad():
                    collated = collate_batch(samples, self.device)
                    output = self.model(collated)
                    result = self.transform_output(output)

                update_predictions_mask(mask, result, collated["meta"])

                if self.debug:
                    break  # process one set of boxes only

        return mask

    def transform_input(self, x: Dict) -> Dict:
        x["img"] = as_channels_last(x["img"])
        return _transform_v2(x, self.config.transform_input)

    def transform_output(self, x: Dict) -> Dict:
        return _transform_v2(x, self.config.transform_output, rename=False)


@dataclass
class MultiModelSlidingWindow(InferenceAlgorithm):
    models: List[pl.LightningModule]
    config: SlidingWindowConfig
    to_segmentation_mask: Callable
    device: torch.device = torch.device("cpu")
    debug: bool = False
    raw_outputs: bool = True

    def __post_init__(self):
        self.models = [
            cast(pl.LightningModule, model.to(self.device)) for model in self.models
        ]

    def predict_from_file(self, filename: str) -> np.ndarray:
        size = self.config.window_size

        with rasterio.open(filename) as dataset:
            h, w = dataset.shape

            boxes = sliding_window_boxes(w, h, size, self.config.overlap)
            n_splits = int(np.ceil(boxes.shape[0] / self.config.max_batch_size))
            mask = np.zeros((h, w), dtype=np.uint8)

            print("boxes widths:", set(boxes[:, 2] - boxes[:, 0]))
            print("boxes heights:", set(boxes[:, 3] - boxes[:, 1]))

            for i, batch in enumerate(np.array_split(boxes, n_splits), 1):
                print(f"processing batch {i} of {n_splits}... ", end="")

                samples = self._read_samples(dataset, batch)

                if samples:
                    print(f"number of samples: {len(samples)}")
                else:
                    print("no relevant samples")
                    continue

                with torch.no_grad():
                    collated = collate_batch(samples, self.device)
                    n_models = len(self.models)
                    outputs = {}

                    print(f"predicting models (total {n_models}): ", end="")

                    for j, model in enumerate(self.models):
                        print(f"{j+1}..", end="")

                        output = model(collated)
                        if not self.raw_outputs:
                            output = self.transform_output(output)
                        outputs[j] = output["outputs"]

                    print("done!")

                    self.to_segmentation_mask(mask, outputs, collated["meta"])

                    if self.debug:
                        break  # process one set of boxes only

        return mask

    def _read_samples(self, dataset: rasterio.DatasetReader, batch: List):
        samples = []

        size = self.config.window_size

        for box in batch:
            x1, y1, x2, y2 = box
            window = Window.from_slices((y1, y2), (x1, x2))
            image = dataset.read(dataset.indexes, window=window)

            if (
                self.config.check_for_outliers and
                outlier(
                    image=image,
                    threshold=self.config.outliers_threshold
                )
            ):
                continue

            samples.append(
                {
                    "box": box,
                    **self.transform_input(
                        {"img": to_rgb(image), "seg": np.zeros((size, size))}
                    ),
                }
            )

        return samples

    def transform_input(self, x: Dict) -> Dict:
        return _transform_v2(x, self.config.transform_input)

    def transform_output(self, x: Dict) -> Dict:
        return (
            x
            if self.config.transform_output is None
            else self.config.transform_output(x)
        )


def collate_batch(samples: List[Dict], device: torch.device = "cpu") -> Dict:
    img = torch.stack([sample["img"] for sample in samples])
    img = img.to(device)
    meta = [
        {k: v for k, v in sample.items() if k not in {"img", "seg"}}
        for sample in samples
    ]
    return {"img": img, "meta": meta}


def update_predictions_mask(
    mask: np.ndarray,
    result: Dict,
    meta: List[Dict],
    interpolation: int = cv.INTER_LINEAR,
):
    for i, info in enumerate(meta):
        x1, y1, x2, y2 = info["box"]
        tensor = result["outputs"]
        if len(meta) > 1:
            tensor = tensor[i]
        array = to_np(tensor.byte())
        height, width = y2 - y1, x2 - x1
        if array.shape[0] != height or array.shape[1] != width:
            array = cv.resize(array, (height, width), interpolation=interpolation)
        mask[y1:y2, x1:x2] = array


def update_predictions_mask_with_majority_vote(
    mask: np.ndarray,
    result: Dict,
    meta: List[Dict],
    interpolation: int = cv.INTER_LINEAR,
    apply_sigmoid: bool = True,
    sigmoid_threshold: float = 0.5,
    min_voters_fraction: float = 0.5,
):
    """Transforms multiple models predictions into a single segmentation mask using majority vote."""

    n_voters = len(result)
    n_min_voters = min(n_voters, max(1, int(n_voters * min_voters_fraction)))

    for i, info in enumerate(meta):
        x1, y1, x2, y2 = info["box"]

        for identifier, tensor in result.items():
            if apply_sigmoid:
                tensor = torch.sigmoid(tensor)
            binary_mask = torch.where(tensor > sigmoid_threshold, 1.0, 0.0)
            array = to_np(binary_mask.byte())
            height, width = y2 - y1, x2 - x1
            if array.shape[0] != height or array.shape[1] != width:
                array = cv.resize(array, (height, width), interpolation=interpolation)
            mask[y1:y2, x1:x2] += array

        mask[y1:y2, x1:x2] = np.where(mask[y1:y2, x1:x2] >= n_min_voters, 1, 0)


def to_rgb(image: np.ndarray) -> np.ndarray:
    if image.shape[0] == 3:
        return image
    elif image.shape[0] == 1:
        return np.repeat(image, 3, 0)
    raise ValueError(f"shape error: {image.shape}")


def _transform_v2(x: Dict, transform: Callable, rename: Optional[Dict] = None) -> Dict:
    """Support both types of transform callable.

    The v2 transformation pipeline doesn't use adapter but forwards params directly.
    """
    out = (
        x if transform is None
        else transform(x) if isinstance(transform, (AlbuAdapter, OutputPostprocessor))
        else transform(image=x["img"], mask=x.get("seg"))
    )
    if rename is None:
        rename = {"image": "img", "mask": "seg"}
    elif not rename:
        return out
    return {rename.get(k, k): v for k, v in out.items()}
