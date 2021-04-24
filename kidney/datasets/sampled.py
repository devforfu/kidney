from abc import ABC
from os.path import join
from typing import Optional, Callable, Tuple

import cv2 as cv
import numpy as np
import zarr
from pydantic import BaseModel
from scipy.interpolate import Rbf
from torch.utils.data.dataset import Dataset

from deepflash import transforms
from kidney.log import get_logger

logger = get_logger(__name__)


class ZarrDataset(Dataset, ABC):
    def __init__(
        self,
        dataset: str,
        tile_shape: Tuple[int, int] = (512, 512),
        padding: Tuple[int, int] = (0, 0),
        scale: int = 1,
    ):
        samples = zarr.open(join(dataset, "samples"))
        labels = zarr.open(join(dataset, "masks", "labels"))
        pdfs = zarr.open(join(dataset, "masks", "pdfs"))

        assert list(samples) == list(labels) == list(pdfs)

        super().__init__()
        self.samples = samples
        self.labels = labels
        self.pdfs = pdfs
        self.keys = list(samples)
        self.tile_shape = tile_shape
        self.padding = padding
        self.scale = scale

    def get_key(self, index: int) -> str:
        return self.keys[index]


class DeformationField:
    def __init__(self, shape: Tuple[int, int], scale: int):
        grid_range = [np.linspace(-d * scale / 2, d * scale / 2 - 1, d) for d in shape]
        xs, ys = np.meshgrid(*grid_range)[::-1]
        self.shape = shape
        self.scale = scale
        self.field = [xs, ys]

    def __call__(
        self,
        data: np.ndarray,
        offset: Tuple[int, int] = (0, 0),
        pad: Tuple[int, int] = (0, 0),
        interpolation: int = cv.INTER_LINEAR,
        border_mode: int = cv.BORDER_REFLECT,
    ):
        out_shape = tuple(int(s - p) for s, p in zip(self.shape, pad))
        coords = [
            np.squeeze(d).astype(np.float32).reshape(out_shape)
            for d in self.get(offset, pad)
        ]

        slices = []
        for i, coord in enumerate(coords):
            c_min, c_max = int(coord.min()), int(coord.max())
            d_max = data.shape[i]
            if c_min < 0:
                c_max = max(-c_min, c_max)
                c_min = 0
            elif c_max > d_max:
                c_min = min(c_min, 2 * d_max - c_max)
                c_max = d_max
                coord -= c_min
            else:
                coord -= c_min
            slices.append(slice(c_min, c_max))

        if len(data.shape) == len(self.shape) + 1:
            tile = np.empty((*out_shape, data.shape[-1]))
            for c in range(data.shape[-1]):
                tile[..., c] = cv.remap(
                    data[slices[0], slices[1], c],  # todo: normalize to 0-1 here or after augmentation pipeline?
                    coords[1],
                    coords[0],
                    interpolation=interpolation,
                    borderMode=border_mode,
                )

        else:
            tile = cv.remap(
                data[slices[0], slices[1]],
                coords[1],
                coords[0],
                interpolation=interpolation,
                borderMode=border_mode,
            )

        return tile

    def rotate(self, angle: float = 0):
        xs, ys = self.field
        new_xs = xs * np.cos(angle) + ys * np.sin(angle)
        new_ys = -xs * np.sin(angle) + ys * np.cos(angle)
        self.field = [new_xs, new_ys]

    def mirror(self, dims: Tuple[int, ...]):
        for i, _ in enumerate(self.shape):
            if i in dims:
                self.field[i] = -self.field[i]

    def add_random_deformation(self, grid: Tuple[int, int], sigma: Tuple[int, int]):
        seed_grid = np.meshgrid(*[
            np.arange(-g/2, s + g/2, g)
            for g, s in zip(grid, self.shape)
        ])
        seed = [
            np.random.normal(0, s, g.shape)
            for g, s in zip(seed_grid, sigma)
        ]
        funcs = [Rbf(*seed_grid, s, function="cubic") for s in seed]
        target_grid = np.meshgrid(*map(np.arange, self.shape))
        deformation = [f(*target_grid) for f in funcs]
        self.field = [f + df for f, df in zip(self.field, deformation)]

    def get(self, offset: Tuple[int, int], pad: Tuple[int, int]):
        index = tuple(slice(int(p / 2), int(-p / 2)) if p > 0 else None for p in pad)
        deformations = [arr[index] for arr in self.field]
        return [d + o for d, o in zip(deformations, offset)]


class DeformationConfig(BaseModel):
    flip: bool = True
    rotation_range: Optional[Tuple[int, int]] = (0, 360)
    grid: Tuple[int, int] = (150, 150)
    magnitude: Tuple[int, int] = (10, 10)
    value_min_range: Tuple[int, int] = (0, 0)
    value_max_range: Tuple[int, int] = (1, 1)
    value_slope_range: Tuple[int, int] = (1, 1)
    zoom_prob: float = 0.75
    zoom_sigma: float = 0.1


class RandomTilesDataset(ZarrDataset):
    def __init__(
        self,
        deformation_config: DeformationConfig = DeformationConfig(),
        samples_per_item: Optional[int] = None,
        **base_params
    ):
        super().__init__(**base_params)

        if samples_per_item is None:
            tile_shape = np.array(self.tile_shape) - np.array(self.padding)
            mask_shape = np.array(self.labels[self.keys[0]].shape[:-1])
            samples_per_item = int(np.product(np.floor(mask_shape / tile_shape)))

        self.samples_per_item = samples_per_item
        self.deformation_config = deformation_config
        self.deformation: DeformationField = None
        self.gamma: Callable = None
        self.update_deformation()

    def __len__(self):
        return len(self.samples) * self.samples_per_item

    def __getitem__(self, item: int):
        key = self.get_key(item % len(self.samples))

        image = self.samples[key]
        mask = self.labels[key]
        pdf = self.pdfs[key]

        center = transforms.random_center(pdf[:], mask.shape)

        x = self.deformation(image, offset=center, pad=self.padding).astype(np.uint8)

        # x = x.flatten().reshape((*self.tile_shape, n_channels))
        # x = self.gamma(x)
        # x = x.transpose(2, 0, 1).astype(np.float32)

        y = self.deformation(mask, offset=center, pad=self.padding, interpolation=cv.INTER_NEAREST)

        return x, y

    def update_deformation(self):
        scale = (
            self.scale * np.random.normal(1, self.deformation_config.zoom_sigma)
            if np.random.random() < self.deformation_config.zoom_prob
            else self.scale
        )

        field = DeformationField(self.tile_shape, scale)

        if self.deformation_config.rotation_range is not None:
            lo, hi = self.deformation_config.rotation_range
            radians = np.pi * (np.random.random() * (hi - lo) + lo) / 180.0
            field.rotate(radians)

        if self.deformation_config.flip:
            field.mirror(dims=np.random.choice([0, 1], 2))

        if self.deformation_config.grid is not None:
            field.add_random_deformation(
                self.deformation_config.grid,
                self.deformation_config.magnitude
            )

        a, b = self.deformation_config.value_min_range
        min_value = a + (b - a)*np.random.random()

        a, b = self.deformation_config.value_max_range
        max_value = a + (b - a)*np.random.random()

        a, b = self.deformation_config.value_slope_range
        inter_value = a + (b - a)*np.random.random()

        # self.gamma = interp1d([0, 0.5, 1.0], [min_value, inter_value, max_value], kind="quadratic")
        self.deformation = field


def main():
    # root = "/mnt/fast/data/kidney/"
    # filenames = read_zarr_files(f"{root}/zarr_train_2")
    # labels_dir, pdfs_dir = create_zarr_files(filenames, output_dir=root, n_jobs=1)
    # dataset = RandomTilesDataset(zarr_files=filenames, labels_dir=labels_dir, pdfs_dir=pdfs_dir)
    # x, y = dataset[0]
    # print(x, y)

    path = "/mnt/fast/data/kidney/zarr/scale_2"
    dataset = RandomTilesDataset(dataset=path)
    print(dataset[0])


if __name__ == "__main__":
    main()
