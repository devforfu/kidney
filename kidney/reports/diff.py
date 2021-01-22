import os
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
from typing import Dict, Tuple

import cv2 as cv
import numpy as np
import pandas as pd
import streamlit as st
from zeus.utils import list_files

from kidney.datasets.kaggle import get_reader
from kidney.reports import session, sidebar, read_image
from kidney.reports.auth import with_password
from kidney.reports.colors import hex_to_color
from kidney.reports.style import set_wide_screen
from kidney.utils.image import overlay_masks
from kidney.utils.mask import rle_decode

session_state = session.get(password=False)

PREDICTIONS_DIR = os.path.join(os.environ["DATASET_ROOT"], "predictions")


def select_model_dir():
    st.subheader("Select predictions")
    model_dir = st.selectbox(label="Model", options=os.listdir(PREDICTIONS_DIR))
    return os.path.join(PREDICTIONS_DIR, model_dir)


@st.cache
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


@dataclass
class CombinedPrediction:
    predictions: Dict
    mask_size: Tuple[int, int]

    def __call__(self, sample_key: str):
        raise NotImplementedError()


class MajorityVotePrediction(CombinedPrediction):
    majority: float = 0.5

    def __call__(self, sample_key: str) -> np.ndarray:
        rle_masks = self.predictions[sample_key]
        n_folds = len(rle_masks)
        majority_threshold = int(self.majority * n_folds)
        mask_pred = np.zeros(self.mask_size, dtype=np.uint8)
        for fold_name, mask in rle_masks.items():
            mask_pred += rle_decode(mask, self.mask_size)
        mask_pred = mask_pred > majority_threshold
        return mask_pred.astype(np.uint8)


@with_password(session_state)
def main():
    set_wide_screen()
    reader = get_reader()
    sample_key, thumb_size = sidebar(reader)

    meta = reader.fetch_meta(sample_key)
    image, info = read_image(meta, thumb_size, overlay_mask=False)
    image_size = info["full_size"]
    rle_df = read_predictions(select_model_dir())
    predictor = MajorityVotePrediction(rle_df.to_dict("index"), image_size)

    mask_pred = predictor(sample_key)

    masks = []

    if st.checkbox(label="Show Prediction", value=True):
        if mask_pred is not None:
            mask_pred = cv.resize(mask_pred, info["thumb_size"])
            color = st.color_picker(label="Prediction mask color", value="#0000ff")
            masks.append((mask_pred, hex_to_color(color)))

    if st.checkbox(label="Show Ground Truth", value=True):
        if info.get("mask") is not None:
            color = st.color_picker(label="Ground Truth mask color", value="#ff0000")
            masks.append((info["mask"], hex_to_color(color)))

    if masks:
        image = overlay_masks(image.copy(), masks, convert_to_uint=False)

    st.image(image, caption="Image with mask(s)")


if __name__ == '__main__':
    main()
