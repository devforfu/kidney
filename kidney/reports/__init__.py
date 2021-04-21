from typing import Dict

import cv2 as cv
import streamlit as st
import tifffile

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
def read_image(meta: Dict, size: int, overlay_mask: bool = True):
    tiff = tifffile.imread(meta["tiff"]).squeeze()
    if tiff.shape[0] == 3:
        tiff = tiff.transpose((1, 2, 0))
    shape = tiff.shape[:2]
    has_mask = meta.get("mask") is not None
    mask = rle_decode(meta["mask"], shape) if has_mask else None
    info = {"thumb_size": (size, size), "full_size": shape}
    if has_mask and overlay_mask:
        thumb = overlay(tiff, mask, alpha=0.4, resize=(size, size))
    elif has_mask:
        thumb = cv.resize(tiff, (size, size))
        mask = cv.resize(mask, (size, size))
        info["mask"] = mask
    else:
        thumb = cv.resize(tiff, (size, size))
    return thumb, info
