import random
from collections import OrderedDict
from functools import reduce
from itertools import chain, product
from typing import List, Dict, Callable, Optional, Set

import numpy as np
from monai.data import list_data_collate
from torch.utils.data import Dataset, DataLoader
from zeus.utils import list_files

from kidney.datasets.kaggle import DatasetReader, SampleType
from kidney.datasets.transformers import Transformers
from kidney.datasets.utils import check_disjoint_subsets, train_valid_keys_split, create_train_valid_data_loaders
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
    valid_keys: Optional[List[str]] = None,
    num_workers: int = 0,
    batch_size: int = 4,
) -> OrderedDict:

    def dataset_factory(name: str, subset_samples: List[Dict]) -> Dataset:
        transform = getattr(transformers, name, None)
        return OnlineCroppingDataset(
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

    # keys = reader.get_keys(SampleType.Labeled)
    # train_keys, valid_keys = train_valid_keys_split(keys, train_keys,  valid_keys)
    # check_disjoint_subsets(set(keys), set(train_keys), set(valid_keys))
    #
    # loaders = OrderedDict()
    # for subset, keys in (
    #     ("train", train_keys),
    #     ("valid", valid_keys)
    # ):
    #     samples_subset = [sample for sample in samples if sample["key"] in keys]
    #     loaders[subset] = DataLoader(
    #         dataset=OnlineCroppingDataset(
    #             reader=reader,
    #             samples=samples_subset,
    #             transform=getattr(transformers, subset, None)
    #         ),
    #         batch_size=batch_size,
    #         shuffle=subset == "train",
    #         num_workers=num_workers,
    #         collate_fn=list_data_collate
    #     )
    # return loaders
