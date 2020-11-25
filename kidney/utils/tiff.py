import gdal
import numpy as np


def read_tiff(path: str) -> np.ndarray:
    """Reads TIFF file."""

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
