from typing import Tuple, Optional

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


def rle_encode(mask: np.ndarray, threshold: Optional[float] = None) -> str:
    """Encoders a binary mask into RLE-string.

    References
    ----------
    [1] https://www.kaggle.com/hackerpoet/even-faster-run-length-encoder

    """

    pixels = mask.T.flatten()

    if threshold is not None:
        pixels = np.where(pixels > threshold, 1, 0)

    pixels = pixels.astype(np.uint8)

    mask_start = (pixels[:-1] == 0) & (pixels[1:] == 1)
    mask_end = (pixels[:-1] == 1) & (pixels[1:] == 0)
    [start_idx] = np.where(mask_start)
    [end_idx] = np.where(mask_end)
    lengths = end_idx - start_idx

    encoded = np.zeros(start_idx.shape[0] + lengths.shape[0])
    encoded[::2] = start_idx + 2  # adjust for counting from 1
    encoded[1::2] = lengths

    return " ".join([str(x) for x in encoded.astype(int)])


def main():
    mask = '9 1 16 4 24 1 39 1 46 1 51 5'
    decoded = rle_decode(mask, (7, 9))
    encoded = rle_encode(decoded)
    assert mask == encoded


if __name__ == '__main__':
    main()
