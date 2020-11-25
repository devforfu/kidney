from typing import Tuple

import numpy as np


def rle_decode(encoded: str, shape: Tuple[int, int]) -> np.ndarray:
    """Decodes an RLE-encoded string."""

    numbers = list(map(int, encoded.split()))
    starts, runs = [np.asarray(xs) for xs in (numbers[::2], numbers[1::2])]

    # pixels enumerations starts from 1 but arrays are
    # indexed staring from 0 so need to make an adjustment
    starts -= 1

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, run in zip(starts, runs):
        mask[start:start + run] = 1

    # In NumPy arrays, first goes height and then goes width; also,
    # the pixels in the mask are enumerated from top to bottom and
    # from left to right, but the mask was filled in a different
    # order so need to transpose
    return mask.reshape(shape[1], shape[0]).T
