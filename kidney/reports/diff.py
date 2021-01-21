import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

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


@st.cache
def select_model_dir():
    st.subheader("Select predictions")
    model_dir = st.selectbox(label="Model", options=os.listdir(PREDICTIONS_DIR))
    return os.path.join(PREDICTIONS_DIR, model_dir)


@st.cache
def read_predictions(root: str) -> Dict[str, Dict[str, str]]:
    pred_df, *dfs = [pd.read_csv(fn).set_index("id") for fn in list_files(root)]
    for df in dfs:
        pred_df = pd.merge(pred_df, df, left_index=True, right_index=True)
    predictions = pred_df.to_dict(orient="index")
    return predictions


@dataclass
class CombinedPrediction:
    predictions: Dict
    mask_size: Tuple[int, int]

    def __call__(self, sample_key: str):
        raise NotImplementedError()


class MajorityVotePrediction(CombinedPrediction):

    def __call__(self, sample_key: str):
        rle_masks = self.predictions[sample_key]
        n_folds = len(rle_masks)
        mask_pred = np.zeros(self.mask_size, dtype=np.uint8)
        for fold_name, mask in rle_masks.items():
            mask_pred += rle_decode(mask, self.mask_size)
        mask_pred = (mask_pred > (n_folds // 2)).astype(np.uint8)
        return mask_pred


@with_password(session_state)
def main():
    set_wide_screen()
    reader = get_reader()
    sample_key, thumb_size = sidebar(reader)

    meta = reader.fetch_meta(sample_key)
    image, info = read_image(meta, thumb_size, overlay_mask=False)
    image_size = image.shape[:2]
    predictor = MajorityVotePrediction(read_predictions(select_model_dir()), image_size)

    mask_pred = predictor(sample_key)

    masks = []

    if st.checkbox(label="Show Prediction", value=True):
        if mask_pred is not None:
            mask_pred = cv.resize(mask_pred, image_size)
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
