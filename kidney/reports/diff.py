import os

import PIL.Image
import numpy as np
import streamlit as st

from kidney.datasets.kaggle import get_reader
from kidney.reports import session, sidebar, read_image
from kidney.reports.auth import with_password
from kidney.reports.colors import hex_to_color
from kidney.reports.style import set_wide_screen
from kidney.utils.image import overlay_masks

session_state = session.get(password=False)


def read_prediction_mask(sample_key: str) -> np.ndarray:
    root = os.path.join(os.environ["DATASET_ROOT"], "predictions")
    mask_path = os.path.join(root, f"{sample_key}.png")
    if not os.path.exists(mask_path):
        return None
    image = PIL.Image.open(mask_path)
    return np.asarray(image)


@with_password(session_state)
def main():
    set_wide_screen()
    reader = get_reader()
    sample_key, thumb_size = sidebar(reader)

    meta = reader.fetch_meta(sample_key)
    image, info = read_image(meta, thumb_size, overlay_mask=False)
    masks = []

    mask_pred = read_prediction_mask(sample_key)
    if mask_pred is not None:
        color = st.color_picker(label="Prediction mask color", value="#0000ff")
        masks.append((mask_pred, hex_to_color(color)))

    if info.get("mask") is not None:
        color = st.color_picker(label="Ground Truth mask color", value="#ff0000")
        masks.append((info["mask"], hex_to_color(color)))

    image_with_overlay = overlay_masks(image.copy(), masks, convert_to_uint=False)
    st.image(image_with_overlay, caption="Image with mask(s)")


if __name__ == '__main__':
    main()
