from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import gdal
import numpy as np


@contextmanager
def capture_gdal_warnings():
    """Suppresses GDAL warnings.

    References:
        1) https://gdal.org/api/python_gotchas.html

    """
    @dataclass
    class GdalErrorHandler:
        err_level: Any = gdal.CE_None
        err_no: int = 0
        err_msg: str = ''

        def handler(self, err_level: Any, err_no: int, err_msg: str):
            self.err_level = err_level
            self.err_no = err_no
            self.err_msg = err_msg

    err = GdalErrorHandler()
    handler = err.handler
    gdal.PushErrorHandler(handler)
    gdal.UseExceptions()
    yield
    gdal.PopErrorHandler()


def read_tiff(path: str) -> np.ndarray:
    """Reads TIFF file."""

    with capture_gdal_warnings():
        dataset = gdal.Open(path, gdal.GA_ReadOnly)

    n_channels = dataset.RasterCount
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    image = np.zeros((n_channels, height, width), dtype=np.uint8)
    for i in range(n_channels):
        band = dataset.GetRasterBand(i + 1)
        channel = band.ReadAsArray()
        image[i] = channel

    return image
