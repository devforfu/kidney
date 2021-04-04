from operator import itemgetter
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from zeus.utils.filesystem import list_files

from kidney.utils.mask import rle_decode


def read_single_prediction(filename: str):
    return pd.read_csv(filename).set_index("id")


def read_predictions(root: str):
    folds = []
    for fn in list_files(root):
        name = Path(fn).stem
        order = int(name.split("_")[-1])
        folds.append((order, fn))

    acc, *rest = [
        pd.read_csv(fn).set_index("id")
        for _, fn in sorted(folds, key=itemgetter(0))
    ]

    for df in rest:
        acc = pd.merge(acc, df, left_index=True, right_index=True)
    acc.columns = range(len(folds))

    return acc


class CombinedPrediction:

    def __init__(self, predictions: Dict, mask_size: Tuple[int, int]):
        self.predictions = predictions
        self.mask_size = mask_size

    def __call__(self, sample_key: str):
        raise NotImplementedError()


class MajorityVotePrediction(CombinedPrediction):

    def __init__(self, majority: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.majority = majority

    def __call__(self, sample_key: str) -> np.ndarray:
        rle_masks = self.predictions[sample_key]
        n_folds = len(rle_masks)
        majority_threshold = int(self.majority * n_folds)
        mask_pred = np.zeros(self.mask_size, dtype=np.uint8)
        for fold_name, mask in rle_masks.items():
            mask_pred += rle_decode(mask, self.mask_size)
        if n_folds > 1:
            mask_pred = mask_pred >= majority_threshold
        return mask_pred.astype(np.uint8)
