from collections import OrderedDict
from typing import List, Dict, Callable, Optional

import numpy as np
from torch.utils.data import Dataset

from kidney.datasets.kaggle import DatasetReader, SampleType
from kidney.datasets.transformers import Transformers
from kidney.datasets.utils import create_train_valid_data_loaders
from kidney.utils.image import pil_read_image
from kidney.utils.mask import rle_decode


class OfflineCroppedDataset(Dataset):

    def __init__(
        self,
        reader: DatasetReader,
        samples: List[Dict],
        transform: Optional[Callable] = None,
        read_image_fn: Callable = pil_read_image
    ):
        super().__init__()
        self.reader = reader
        self.samples = samples
        self.transform = transform
        self.read_image_fn = read_image_fn

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item: int) -> Dict:
        sample = self.samples[item]
        meta = self.reader.fetch_meta(sample["key"])
        x1, y1, x2, y2 = sample["box"]
        mask_crop = rle_decode(sample["rle_encoded"], shape=(y2 - y1, x2 - x1))
        image = self.read_image_fn(meta["png"])
        sample = {"img": image.astype(np.float32), "seg": mask_crop.astype(np.float32)}
        return sample if self.transform is None else self.transform(sample)


def create_data_loaders(
    reader: DatasetReader,
    transformers: Transformers,
    samples: List[Dict],
    train_keys: Optional[List[str]] = None,
    valid_keys: Optional[List[str]] = None,
    num_workers: int = 0,
    batch_size: int = 4,
) -> OrderedDict:

    def dataset_factory(name: str, subset_samples: List[Dict]) -> Dataset:
        transform = getattr(transformers, name, None)
        return OfflineCroppedDataset(
            reader=reader,
            samples=subset_samples,
            transform=transform
        )

    return create_train_valid_data_loaders(
        keys=reader.get_keys(SampleType.Labeled),
        transformers=transformers,
        dataset_factory=dataset_factory,
        samples=samples,
        train_keys=train_keys,
        valid_keys=valid_keys,
        num_workers=num_workers,
        batch_size=batch_size
    )