from typing import Dict, Tuple

import cv2 as cv
import numpy as np
import streamlit as st

from kidney.datasets.kaggle import get_reader, SampleType, DatasetReader
from kidney.reports.auth import with_password
from kidney.reports import session
from kidney.reports.style import wide_screen_style
from kidney.utils.image import overlay
from kidney.utils.mask import rle_decode
from kidney.utils.tiff import read_tiff, read_tiff_crop

session_state = session.get(password=False)


def sidebar(reader: DatasetReader):
    sample_type = SampleType[
        st.sidebar.selectbox(
            label="Sample type",
            options=[x.name for x in SampleType]
        )
    ]
    keys = reader.get_keys(sample_type)
    selected_key = st.sidebar.selectbox("Image key", keys)
    size = st.sidebar.radio("Thumbnail size", options=[4096, 2048, 1024, 512])
    return selected_key, size


@st.cache
def read_image(meta: Dict, size: int):
    tiff = read_tiff(meta["tiff"])
    shape = tiff.shape[:2]
    mask = rle_decode(meta["mask"], shape)
    thumb = overlay(tiff, mask, alpha=0.4, resize=(size, size))
    return thumb, {"small": (size, size), "large": shape}


def rescale(
    top_left: Tuple[int, int],
    full_size: Tuple[int, int],
    thumb_size: Tuple[int, int]
) -> Tuple[int, int]:
    x, y = top_left
    full_h, full_w = full_size
    thumb_h, thumb_w = thumb_size
    new_x = int(x * thumb_w/full_w)
    new_y = int(y * thumb_h/full_h)
    return new_x, new_y


@with_password(session_state)
def main():
    st.markdown(wide_screen_style(1600), unsafe_allow_html=True)

    reader = get_reader()
    sample_key, thumb_size = sidebar(reader)
    meta = reader.fetch_meta(sample_key)

    st.header("Image Preview")
    image, sizes = read_image(meta, thumb_size)

    zoom = st.selectbox(label="zoom area size", options=[4096, 2048, 1024, 512, 256])
    zoom_thumb = st.selectbox(label="zoom thumbnail size", options=[1024, 512])
    show_mask = st.checkbox(label="Show mask", value=True)
    full_h, full_w = sizes["large"]
    x = st.slider(label="x", min_value=0, max_value=full_w - zoom)
    y = st.slider(label="y", min_value=0, max_value=full_h - zoom)

    new_x, new_y = rescale(
        top_left=(x, y),
        full_size=sizes["large"],
        thumb_size=sizes["small"]
    )
    thumb_h, thumb_w = sizes["small"]
    # zoom_relative = int(zoom * max(thumb_w/full_w, thumb_h/full_h))
    zoom_x = int(zoom * thumb_w/full_w)
    zoom_y = int(zoom * thumb_h/full_h)
    new_image = cv.rectangle(
        image.copy(),
        (new_x, new_y),
        (new_x + zoom_x, new_y + zoom_y),
        (255, 165, 0),
        thickness=10,
    )
    small = read_tiff_crop(meta["tiff"], (x, y, x + zoom, y + zoom))
    small = np.moveaxis(small, 0, -1)
    if show_mask:
        mask = rle_decode(meta["mask"], (full_h, full_w))
        mask = mask[y:y+zoom, x:x+zoom]
        small = overlay(small, mask, resize=(zoom_thumb, zoom_thumb))
    else:
        small = cv.resize(small, (zoom_thumb, zoom_thumb))
    st.image(small, caption="zoomed section")
    st.image(new_image, caption=sample_key)


if __name__ == '__main__':
    main()
