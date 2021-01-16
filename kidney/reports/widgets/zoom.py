from dataclasses import dataclass
from typing import Dict, Tuple

import cv2 as cv
import numpy as np
import streamlit as st

from kidney.utils.image import overlay
from kidney.utils.mask import rle_decode
from kidney.utils.tiff import read_tiff_crop


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


@dataclass
class ZoomOptions:
    x: int
    y: int
    crop_size: int
    thumb_size: int
    show_mask: bool

    @property
    def thumb_size_tuple(self) -> Tuple[int, int]:
        return self.thumb_size, self.thumb_size


class ZoomController:

    def __init__(self, image: np.ndarray, image_info: Dict):
        self.image = image
        self.image_info = image_info
        self.full_h, self.full_w = self.full_size = self.image_info["full_size"]
        self.thumb_h, self.thumb_w = self.thumb_size = self.image_info["thumb_size"]
        self._zoom: ZoomOptions = None

    def set_zoomed_area(self):
        crop_size = st.selectbox(label="zoom crop area size", options=[256, 512, 1024, 2048, 4096])
        thumb_size = st.selectbox(label="zoom thumbnail size", options=[256, 512, 1024])
        show_mask = st.checkbox(label="Show mask", value=True)
        x = st.slider(label="x", min_value=0, max_value=self.full_w - crop_size)
        y = st.slider(label="y", min_value=0, max_value=self.full_h - crop_size)
        self._zoom = ZoomOptions(x, y, crop_size, thumb_size, show_mask)

    def render_zoom_selection(self, image: np.ndarray, caption: str = "") -> np.ndarray:
        zoom = self._zoom
        if zoom is None:
            return

        new_x, new_y = rescale(
            top_left=(zoom.x, zoom.y),
            full_size=self.full_size,
            thumb_size=self.thumb_size
        )
        zoom_x = int(zoom.crop_size * self.thumb_w/self.full_w)
        zoom_y = int(zoom.crop_size * self.thumb_h/self.full_h)
        image_with_selection = cv.rectangle(
            image.copy(),
            (new_x, new_y),
            (new_x + zoom_x, new_y + zoom_y),
            (255, 165, 0),
            thickness=10
        )
        st.image(image_with_selection, caption=caption)

    def render_selected_area(self, filename: str, rle_mask: str = None):
        x, y, crop_size = self._zoom.x, self._zoom.y, self._zoom.crop_size
        small = read_tiff_crop(filename, (x, y, x + crop_size, y + crop_size))
        small = np.moveaxis(small, 0, -1)
        if self._zoom.show_mask and rle_mask is not None:
            mask = rle_decode(rle_mask, self.full_size)
            mask = mask[y:y + crop_size, x:x + crop_size]
            small = overlay(small, mask, resize=self._zoom.thumb_size_tuple)
        else:
            small = cv.resize(small, self._zoom.thumb_size)
        st.image(small, caption="zoomed section")

