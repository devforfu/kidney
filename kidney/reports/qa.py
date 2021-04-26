import os
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, Any

import cv2 as cv
import numpy as np
import pandas as pd
import streamlit as st
import torch
from monai.metrics import DiceMetric

from kidney.datasets.kaggle import KaggleKidneyDatasetReader
from kidney.inference.prediction import MajorityVotePrediction
from kidney.reports import session, sidebar, read_image
from kidney.reports.auth import with_password
from kidney.reports.colors import hex_to_color
from kidney.reports.style import set_wide_screen
from kidney.utils.image import overlay_masks
from kidney.utils.mask import rle_decode

session_state = session.get(password=False)


@with_password(session_state)
def main():
    set_wide_screen()

    paths = Paths.from_env()
    reader = KaggleKidneyDatasetReader(paths.training_images)
    sample_key, thumb_size = sidebar(reader)

    meta = reader.fetch_meta(sample_key)
    image, info = read_image(meta, thumb_size, overlay_mask=False)
    image_size = info["full_size"]
    predictions = pick_model(paths.get_models())

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

    st.text(f"Dice metric: {dice_metric(mask_pred, meta['mask']):2.4f}")

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


@dataclass
class Paths:
    root: str

    @staticmethod
    def from_env() -> "Paths":
        data_root = os.environ.get("DATA_ROOT")
        if data_root is None:
            return None
        return Paths(data_root)

    @property
    def training_images(self) -> str:
        return os.path.join(self.root, "raw")

    @property
    def predicted_masks_dir(self) -> str:
        return os.path.join(self.root, "outputs")

    def get_models(self):
        discovered_models = {}

        def walk(root: str):
            nonlocal discovered_models
            for dirpath, dirnames, filenames in os.walk(root):
                if dirnames:
                    for name in dirnames:
                        walk(name)
                for filename in filenames:
                    if not filename.endswith(".csv"):
                        continue
                    path = os.path.join(dirpath, filename)
                    if fnmatch(filename, "fold_*.csv"):
                        identifier = os.path.basename(dirpath)
                        if identifier not in discovered_models:
                            discovered_models[identifier] = []
                        discovered_models[identifier].append(path)
                    else:
                        identifier = Path(filename).stem
                        discovered_models[identifier] = path

        walk(self.predicted_masks_dir)
        return discovered_models


def pick_model(discovered_models: Dict[str, Any]):
    identifier = st.selectbox(
        label="Model identifier",
        options=list(discovered_models.keys())
    )
    return read_model(discovered_models, identifier)


@st.cache
def read_model(discovered_models: Dict[str, Any], identifier: str):
    path = discovered_models[identifier]
    if isinstance(path, str):
        df = pd.read_csv(path).set_index("id")
    else:
        folds = []
        for fn in path:
            name = Path(fn).stem
            key = name.replace("fold_", "")
            folds.append((key, fn))
        dfs = [
            pd.read_csv(fn).set_index("id")
            for _, fn in sorted(folds)
        ]
        if len(dfs) == 1:
            [df] = dfs
        else:
            acc, *rest = dfs
            for df in rest:
                acc = pd.merge(acc, df, left_index=True, right_index=True)
            acc.columns = range(len(folds))
            df = acc
    return df.to_dict("index")


@st.cache
def dice_metric(mask_pred: np.ndarray, mask_encoded: str) -> float:
    dice = DiceMetric()
    mask_true = rle_decode(mask_encoded, mask_pred.shape)
    dice_value = dice(
        y_pred=torch.tensor(mask_pred).unsqueeze(0),
        y=torch.tensor(mask_true).unsqueeze(0)
    )
    return dice_value.item()


if __name__ == '__main__':
    main()
