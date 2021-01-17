import os

import cv2 as cv
import numpy as np
import streamlit as st

from kidney.datasets.kaggle import get_reader
from kidney.reports import session, sidebar, read_image
from kidney.reports.auth import with_password
from kidney.reports.colors import hex_to_color
from kidney.reports.style import set_wide_screen
from kidney.utils.image import overlay_masks

session_state = session.get(password=False)

PREDICTIONS_DIR = os.path.join(os.environ["DATASET_ROOT"], "predictions")


def select_model_dir():
    st.subheader("Select predictions")
    model_dir = st.selectbox(label="Model", options=os.listdir(PREDICTIONS_DIR))
    return os.path.join(PREDICTIONS_DIR, model_dir)


def read_prediction_mask(root: str, sample_key: str) -> np.ndarray:
    mask_path = os.path.join(root, f"{sample_key}.png")
    if not os.path.exists(mask_path):
        return None
    return (cv.imread(mask_path, 0) == 255).astype(np.uint8)


@with_password(session_state)
def main():
    set_wide_screen()
    reader = get_reader()
    sample_key, thumb_size = sidebar(reader)

    meta = reader.fetch_meta(sample_key)
    image, info = read_image(meta, thumb_size, overlay_mask=False)

    masks = []

    mask_pred = read_prediction_mask(select_model_dir(), sample_key)

    if st.checkbox(label="Show Prediction", value=True):
        if mask_pred is not None:
            mask_pred = cv.resize(mask_pred, image.shape[:2])
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
