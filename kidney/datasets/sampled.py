from abc import ABC
from os.path import join
from typing import Optional, Callable, Tuple, List, Dict

import cv2 as cv
import numpy as np
import torch
import zarr
from pydantic import BaseModel
from scipy.interpolate import Rbf
from torch.utils.data.dataset import Dataset
from zeus.plotting.utils import axes

from deepflash import transforms
from kidney.datasets.offline import float32
from kidney.log import get_logger

logger = get_logger(__name__)


class ZarrDataset(Dataset, ABC):
    _default_keys_mapping = {"image": "img", "mask": "seg"}

    def __init__(
        self,
        path: str,
        subset: str = "samples",
        keys: Optional[List[str]] = None,
        tile_shape: Tuple[int, int] = (512, 512),
        padding: Tuple[int, int] = (0, 0),
        scale: int = 1,
        keys_mapping: Optional[Dict] = None,
    ):
        samples = zarr.open(join(path, subset))
        labels = zarr.open(join(path, "masks", "labels"))
        pdfs = zarr.open(join(path, "masks", "pdfs"))
        keys = keys or list(samples)

        if subset == "samples":
            assert list(samples) == list(labels) == list(pdfs)
            assert len(keys) == len(set(keys))

        assert set(keys).issubset(samples)

        super().__init__()
        self.samples = samples
        self.labels = labels
        self.pdfs = pdfs
        self.keys = keys
        self.tile_shape = tile_shape
        self.padding = padding
        self.scale = scale
        self.keys_mapping = keys_mapping or self._default_keys_mapping

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
                    # todo: normalize to 0-1 here or after augmentation pipeline?
                    data[slices[0], slices[1], c],
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
        samples_per_item: Optional[int] = None,
        transform: Optional[Callable] = None,
        deformation_config: DeformationConfig = DeformationConfig(),
        **base_params
    ):
        super().__init__(**base_params)

        if samples_per_item is None:
            tile_shape = np.array(self.tile_shape) - np.array(self.padding)
            mask_shape = np.array(self.labels[self.keys[0]].shape[:-1])
            samples_per_item = int(np.product(np.floor(mask_shape / tile_shape)))

        self.samples_per_item = samples_per_item
        self.deformation_config = deformation_config
        self.transform = transform
        self.deformation: DeformationField = None
        self.gamma: Callable = None
        self.update_deformation()

    def __len__(self) -> int:
        return len(self.keys) * self.samples_per_item

    def __getitem__(self, item: int) -> Dict:
        key = self.get_key(item % len(self.keys))

        image = self.samples[key]
        mask = self.labels[key]
        pdf = self.pdfs[key]

        center = transforms.random_center(pdf[:], mask.shape)
        x = self.deformation(image, center, self.padding)
        y = self.deformation(mask, center, self.padding, interpolation=cv.INTER_NEAREST)

        if self.transform is None:
            x, y = float32(x), float32(y)
        else:
            transformed = self.transform(image=x.astype(np.uint8), mask=y)
            transformed = {self.keys_mapping.get(k, k): v for k, v in transformed.items()}
            x, y = transformed["img"], transformed["seg"]
            if torch.is_tensor(x):
                x, y = x.float(), y.float()
            else:
                x, y = float32(x), float32(y)

        return {"img": x, "seg": y}

        # if self.transform is None:
        #     return {"img": float32(image), "seg": float32(mask)}
        # else:
        #     x = self.deformation(image, offset=center, pad=self.padding).astype(np.uint8)
        #     y = self.deformation(mask, offset=center, pad=self.padding, interpolation=cv.INTER_NEAREST)
        #     return x, y

        # x = x.flatten().reshape((*self.tile_shape, n_channels))
        # x = self.gamma(x)
        # x = x.transpose(2, 0, 1).astype(np.float32)

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

        # xs = [0, 0.5, 1.0]
        # ys = [min_value, inter_value, max_value]
        # self.gamma = interp1d(xs, ys, kind="quadratic")

        self.deformation = field


class TileDataset(ZarrDataset):

    def __init__(self, transform: Optional[Callable] = None, include_meta: bool = False, **base_params):
        super().__init__(**base_params)
        self.transform = transform
        self.include_meta = include_meta
        self.deformation = DeformationField(self.tile_shape, self.scale)
        self._create_tiles()

    def _create_tiles(self):
        output_shape = tuple(int(t - p) for t, p in zip(self.tile_shape, self.padding))
        centers, indices, shapes, out_slices, in_slices = [], [], [], [], []
        total = 0

        for i, key in enumerate(self.keys):
            image = self.samples[key]
            shape = tuple(int(x//self.scale) for x in image.shape[:-1])

            y_steps = int(max(1, np.ceil(shape[0]/output_shape[0])))
            x_steps = int(max(1, np.ceil(shape[1]/output_shape[1])))

            for y in range(y_steps):
                for x in range(x_steps):
                    cy = int((y + .5)*output_shape[0]*self.scale)
                    cx = int((x + .5)*output_shape[1]*self.scale)

                    centers.append((cy, cx))
                    indices.append(i)
                    shapes.append(shape)

                    slices = [
                        slice(
                            int(pos*out_val),
                            int(min((pos + 1)*out_val, in_val))
                        )
                        for pos, out_val, in_val in zip((y, x), output_shape, shape)
                    ]
                    out_slices.append(slices)

                    slices = [
                        slice(
                            0,
                            int(min((pos + 1)*out_val, in_val) - pos*out_val)
                        )
                        for pos, out_val, in_val in zip((y, x), output_shape, shape)
                    ]
                    in_slices.append(slices)

                    total += 1

        self.centers = centers
        self.indices = indices
        self.shapes = shapes
        self.out_slices = out_slices
        self.in_slices = in_slices
        self.total = total

    def full_size(self, item: int):
        return self.samples[self.keys[self.indices[item]]].shape

    def __len__(self):
        return self.total

    def __getitem__(self, item: int):
        idx = self.indices[item]

        image = self.samples[self.keys[idx]]
        mask = self.labels.get(self.keys[idx])
        center = self.centers[item]

        x = self.deformation(image, center)
        if mask is not None:
            y = self.deformation(mask, center)
        else:
            y = np.zeros(x.shape[:2], dtype=np.uint8)

        if self.transform is None:
            sample = {"img": float32(x), "seg": float32(y)}
        else:
            transformed = self.transform(image=x.astype(np.uint8), mask=y)
            transformed = {self.keys_mapping.get(k, k): v for k, v in transformed.items()}
            image, mask = transformed["img"], transformed["seg"]
            if torch.is_tensor(image):
                image, mask = image.float(), mask.float()
            else:
                image, mask = float32(image), float32(mask)
            sample = {"img": image, "seg": mask}

        if self.include_meta:
            sample["item"] = item
            sample["idx"] = idx

        return sample


def show(
    ds: Dataset,
    n: int,
    m: int,
    size: int = 2,
    random: bool = True,
    indices: List[int] = None
):
    canvas = axes(subplots=(n, m), figsize=(m * size, n * size))

    if random:
        indices = np.random.choice(np.arange(len(ds)), size=m)

        for i in range(n):

            for j, index in enumerate(indices):
                ax = canvas[i][j]
                sample = ds[index]
                x, y = sample["img"], sample["seg"]
                ax.imshow(x.astype(np.uint8))
                ax.imshow(y, alpha=0.3)
                ax.set_axis_off()
                ax.set_title(f"{j} ({x.mean().round(1):.1f})")

            if hasattr(ds, "update_deformation"):
                ds.update_deformation()

    else:
        assert indices is not None

        for i in range(n * m):
            ax = canvas.flat[i]
            sample = ds[i]
            x, y = sample["img"], sample["seg"]
            ax.imshow(x.astype(np.uint8))
            ax.imshow(y, alpha=0.3)
            ax.set_axis_off()
            ax.set_title(i)
