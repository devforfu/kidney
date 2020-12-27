import random
from collections import OrderedDict
from itertools import chain
from typing import List, Dict, Callable, Optional

import numpy as np
from monai.data import list_data_collate
from torch.utils.data import Dataset, DataLoader
from zeus.utils import list_files

from kidney.datasets.kaggle import DatasetReader, SampleType
from kidney.datasets.transformers import Transformers
from kidney.utils import rle
from kidney.utils.mask import rle_decode
from kidney.utils.tiff import read_tiff_crop


class OnlineCroppingDataset(Dataset):

    def __init__(
        self,
        reader: DatasetReader,
        samples: List[Dict],
        transform: Optional[Callable] = None
    ):
        super().__init__()
        self.reader = reader
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item: int) -> Dict:
        sample = self.samples[item]
        meta = self.reader.fetch_meta(sample["key"])
        x1, y1, x2, y2 = box = sample["box"]
        mask_crop = rle_decode(sample["rle_encoded"], shape=(y2 - y1, x2 - x1))
        image = read_tiff_crop(meta["tiff"], box)
        sample = {"img": image.astype(np.float32), "seg": mask_crop.astype(np.float32)}
        return sample if self.transform is None else self.transform(sample)


def create_data_loaders(
    reader: DatasetReader,
    transformers: Transformers,
    samples: List[Dict],
    train_keys: Optional[List[str]] = None,
    num_workers: int = 0,
    batch_size: int = 4,
) -> OrderedDict:

    keys = reader.get_keys(SampleType.Labeled)
    if train_keys is None:
        valid_keys = [random.choice(keys)]
        train_keys = [key for key in keys if key not in valid_keys]
    else:
        valid_keys = [key for key in keys if key not in train_keys]

    loaders = OrderedDict()
    for subset, keys in (
        ("train", train_keys),
        ("valid", valid_keys)
    ):
        samples_subset = [sample for sample in samples if sample["key"] in keys]
        loaders[subset] = DataLoader(
            dataset=OnlineCroppingDataset(
                reader=reader,
                samples=samples_subset,
                transform=getattr(transformers, subset, None)
            ),
            batch_size=batch_size,
            shuffle=subset == "train",
            num_workers=num_workers,
            collate_fn=list_data_collate
        )
    return loaders


def read_boxes(folder: str) -> List[Dict]:
    """Reads pre-generated sliding window boxes from JSONL files."""

    import srsly
    return list(chain(*[srsly.read_jsonl(fn) for fn in list_files(folder)]))
