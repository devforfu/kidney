from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Tuple

import gdal
import numpy as np
import rasterio
from rasterio.windows import Window
from tifffile import tifffile


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

    return image.transpose((1, 2, 0))


def read_tiff_all(path: str) -> np.ndarray:
    img = tifffile.imread(path).squeeze()
    if img.shape[0] == 3:
        img = img.transpose((1, 2, 0))
    return img


def read_tiff_crop(path: str, box: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
    with rasterio.open(path, transform=identity) as dataset:
        window = Window.from_slices((y1, y2), (x1, x2))
        crop = dataset.read(dataset.indexes, window=window)
    return crop
