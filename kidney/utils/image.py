from typing import Tuple

import cv2 as cv
import numpy as np


def detect_roi(
    image: np.ndarray,
    thumbnail_size: Tuple[int, int] = (1024, 1024)
) -> Tuple[int, int, int, int]:
    """Detects the biggest contour on the image.

    Helpful to get rid of non-informative boundary around the training images.

    Args:
        image: Training image to cut the boundaries.
        thumbnail_size: Size to reduce the image before running contours detector.

    Returns:
        Tuple: The detected box.

    """
    thumbnail = cv.resize(image, thumbnail_size)
    gray = cv.cvtColor(thumbnail, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_OTSU + cv.THRESH_BINARY)[1]

    contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) else contours[1]
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    biggest = contours[0]

    xr, yr, wr, hr = cv.boundingRect(biggest)
    w0, h0 = image.shape[1], image.shape[0]
    wt, ht = thumbnail_size
    rw, rh = w0/wt, h0/ht
    x0, y0, x1, y1 = [int(v) for v in [
        (xr * rw),
        (yr * rh),
        (xr + wr) * rw,
        (yr + hr) * rh
    ]]

    return x0, y0, x1, y1
