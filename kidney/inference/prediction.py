from typing import Dict, Tuple

import numpy as np

from kidney.utils.mask import rle_decode


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
        mask_pred = mask_pred >= majority_threshold
        return mask_pred.astype(np.uint8)
