import os
from os.path import join

import cv2 as cv
import streamlit as st

from kidney.datasets.kaggle import KaggleKidneyDatasetReader
from kidney.inference.prediction import MajorityVotePrediction
from kidney.reports import session, sidebar, read_image
from kidney.reports.auth import with_password
from kidney.reports.colors import hex_to_color
from kidney.reports.style import set_wide_screen
from kidney.utils.image import overlay_masks

session_state = session.get(password=False)


@with_password(session_state)
def main():
    set_wide_screen()
    root = get_root()
    reader = KaggleKidneyDatasetReader(join(root, "raw"))
    sample_key, thumb_size = sidebar(reader)
    path, is_dir = select_predictions(root)

    try:
        predictions = read_predictions(path) if is_dir else read_single_prediction(path)
    except Exception as e:
        st.error(f"Cannot read predictions from location: {path}")
        st.error(str(e))
        return

    meta = reader.fetch_meta(sample_key)
    image, info = read_image(meta, thumb_size, overlay_mask=False)
    image_size = info["full_size"]

    predictor = MajorityVotePrediction(
        predictions=predictions,
        mask_size=image_size,
        majority=st.slider(
            label="Majority threshold",
            min_value=0.0, max_value=1.0,
            value=0.5, step=0.25
        )
    )

    mask_pred = predictor(sample_key)

    masks = []

    if st.checkbox(label="Show Prediction", value=True):
        if mask_pred is not None:
            mask_pred = cv.resize(mask_pred, info["thumb_size"])
            color = st.color_picker(label="Prediction mask color", value="#00ff00")
            masks.append((mask_pred, hex_to_color(color)))

    if st.checkbox(label="Show Ground Truth", value=True):
        if info.get("mask") is not None:
            color = st.color_picker(label="Ground Truth mask color", value="#ff0000")
            masks.append((info["mask"], hex_to_color(color)))

    if st.checkbox(label="Show Combined Masks", value=len(masks) == 2):
        color = st.color_picker(label="Intersection color", value="#ffff00")
        pred, gt = [mask for mask, color in masks]
        combined = pred & gt
        masks.append((combined, hex_to_color(color)))

    if masks:
        image = overlay_masks(image.copy(), masks, convert_to_uint=False)

    st.image(image, caption="Image with mask(s)")



def get_root():
    return os.environ.get("DATASET_ROOT", os.path.expanduser("~"))


def select_predictions(root: str):
    choices = []
    for fn in os.listdir(root):
        for fn2 in os.listdir(join(root, fn)):
            choices.append(join(root, fn, fn2))
    filename = st.selectbox("Select predictions", choices)
    return filename, os.path.isdir(filename)


@st.cache
def read_predictions(dirname: str):
    from kidney.inference.prediction import read_predictions
    return read_predictions(dirname).to_dict("index")


@st.cache
def read_single_prediction(filename: str):
    from kidney.inference.prediction import read_single_prediction
    return read_single_prediction(filename).to_dict("index")



if __name__ == '__main__':
    main()
