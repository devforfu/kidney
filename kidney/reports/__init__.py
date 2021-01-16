from typing import Dict

import cv2 as cv
import streamlit as st

from kidney.datasets.kaggle import DatasetReader, SampleType
from kidney.utils.image import overlay
from kidney.utils.mask import rle_decode
from kidney.utils.tiff import read_tiff


def sidebar(reader: DatasetReader):
    sample_type = SampleType[
        st.sidebar.selectbox(
            label="Sample type",
            options=[x.name for x in SampleType]
        )
    ]
    keys = reader.get_keys(sample_type)
    selected_key = st.sidebar.selectbox("Image key", keys)
    size = st.sidebar.radio("Thumbnail size", options=[512, 1024, 2048, 4096])
    return selected_key, size


@st.cache
def read_image(meta: Dict, size: int):
    tiff = read_tiff(meta["tiff"])
    shape = tiff.shape[:2]
    if meta.get("mask") is not None:
        mask = rle_decode(meta["mask"], shape)
        thumb = overlay(tiff, mask, alpha=0.4, resize=(size, size))
    else:
        thumb = cv.resize(tiff, (size, size))
    return thumb, {"thumb_size": (size, size), "full_size": shape}
