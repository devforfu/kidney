from dataclasses import dataclass
from typing import Dict, cast

import numpy as np
import pytorch_lightning as pl
import tifffile
import torch

from kidney.datasets.kaggle import outlier
from kidney.datasets.transformers import as_channels_last
from kidney.inference.inference import _transform_v2, update_predictions_mask, collate_batch, to_rgb, InferenceAlgorithm
from kidney.inference.window import sliding_window_boxes
from prototype.config import SlidingWindowConfig


@dataclass
class InMemorySlidingWindow(InferenceAlgorithm):
    model: pl.LightningModule
    config: SlidingWindowConfig
    device: torch.device = torch.device("cpu")
    debug: bool = False

    def __post_init__(self):
        self.model = cast(pl.LightningModule, self.model.to(self.device))

    def predict_from_file(self, filename: str) -> np.ndarray:
        size = self.config.window_size

        print("input image size:", size)
        if self.config.check_for_outliers:
            print("outliers threshold:", self.config.outliers_threshold)

        large_image = tifffile.imread(filename).squeeze()
        if large_image.shape[-1] == 3:
            large_image = large_image.transpose((2, 0, 1))

        print("image shape:",  large_image.shape)

        h, w = large_image.shape[-2:]
        boxes = sliding_window_boxes(w, h, size, self.config.overlap)
        n_splits = int(np.ceil(boxes.shape[0] / self.config.max_batch_size))
        mask = np.zeros((h, w), dtype=np.uint8)

        print("boxes widths:", set(boxes[:, 2] - boxes[:, 0]))
        print("boxes heights:", set(boxes[:, 3] - boxes[:, 1]))
        print("the number of boxes:", len(boxes))

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
                        large_image[:, y1:y2, x1:x2]
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
