import os

import PIL.Image
import numpy as np
import streamlit as st

from kidney.datasets.kaggle import get_reader
from kidney.reports import session, sidebar, read_image
from kidney.reports.auth import with_password
from kidney.reports.style import set_wide_screen
from kidney.utils.image import overlay_masks

session_state = session.get(password=False)

MASK_COLOR_GT = (255, 0, 0)
MASK_COLOR_PRED = (0, 0, 255)


def read_prediction_mask(sample_key: str) -> np.ndarray:
    root = os.environ.get("PREDICTIONS_DIR", "predictions")
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

    masks = []
    mask_pred = read_prediction_mask(sample_key)
    if mask_pred is not None:
        masks.append((mask_pred, MASK_COLOR_PRED))

    meta = reader.fetch_meta(sample_key)
    image, info = read_image(meta, thumb_size, overlay_mask=False)
    if info.get("mask") is not None:
        masks.append((info["mask"], MASK_COLOR_GT))

    image_with_overlay = overlay_masks(image, masks, convert_to_uint=False)
    st.image(image_with_overlay, caption="Image with mask(s)")


if __name__ == '__main__':
    main()
