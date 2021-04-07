import json
import random

import cv2 as cv
import numpy as np
from zeus.utils import list_files


class ColorTransfer:

    def __init__(self, mean: np.ndarray, std: np.ndarray, ref: str = "default"):
        self.mean = mean
        self.std = std
        self.ref = ref

    @staticmethod
    def read_json(filename: str):
        with open(filename, "r") as fp:
            contents = json.load(fp)
        mean = [np.array(c) for c in contents["mean"]]
        std = [np.array(c) for c in contents["std"]]
        return ColorTransfer(mean, std)

    def write_json(self, filename: str):
        with open(filename, "w") as fp:
            json.dump({
                "mean": [c.tolist() for c in self.mean],
                "std": [c.tolist() for c in self.std],
                "reference": self.ref
            }, fp)

    def transfer_image(self, target: np.ndarray, as_rgb: bool = True):
        channels = []
        for i, channel in enumerate(cv.split(target)):
            channel -= channel.mean()
            channel *= channel.std() / (self.std[i] + 1e-8)
            channel += self.mean[i]
            channel = channel.clip(0, 255)
            channels.append(channel)
        image = cv.merge(channels).astype(np.uint8)
        if as_rgb:
            image = cv.cvtColor(image, cv.COLOR_LAB2RGB)
        return image


def read_lab(filename: str):
    bgr = cv.imread(filename)
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB).astype(np.float32)
    return lab


def channel_stats(image: np.ndarray):
    channels = cv.split(image)
    mean = [c.mean() for c in channels]
    std = [c.std() for c in channels]
    return mean, std


class ColorTransferAugmentation:

    def __init__(self, stats_dir: str, prob: float = 0.5):
        self.transfers = [ColorTransfer.read_json(fn) for fn in list_files(stats_dir)]
        self.prob = prob

    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        if random.random() < (1 - self.prob):
            return image
        transfer = random.choice(self.transfers)
        return transfer.transfer_image(image)
