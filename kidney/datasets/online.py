import random
from collections import OrderedDict
from typing import List, Dict, Tuple, Callable, Optional

import numpy as np
from monai.data import list_data_collate
from torch.utils.data import Dataset, DataLoader

from kidney.datasets.kaggle import DatasetReader, SampleType, get_reader, outlier
from kidney.datasets.transformers import Transformers
from kidney.inference.window import SlidingWindowsGenerator
from kidney.utils.mask import rle_decode
from kidney.utils.tiff import read_tiff_crop


class OnlineCroppingDataset(Dataset):

    def __init__(
        self,
        keys: List[str],
        reader: DatasetReader,
        sliding_windows_generator: SlidingWindowsGenerator,
        outliers_excluded: bool = True,
        outliers_threshold: int = 1000,
        transform: Optional[Callable] = None
    ):
        self.keys = keys
        self.reader = reader
        self.sliding_windows_generator = sliding_windows_generator
        self.samples = self._generate_samples(outliers_excluded, outliers_threshold)
        self.transform = transform

    def _generate_samples(self, exclude_outliers: bool, threshold: int) -> List[Tuple]:
        generated = []
        for key in self.keys:
            meta = self.reader.fetch_meta(key)
            filename = meta["tiff"]
            boxes, (h, w) = self.sliding_windows_generator.generate(filename)
            mask = rle_decode(meta["mask"], shape=(h, w))
            for box in boxes:
                x1, y1, x2, y2 = box
                if exclude_outliers:
                    crop = read_tiff_crop(filename, box)
                    if outlier(crop, threshold=threshold):
                        continue
                mask_crop = mask[y1:y2, x1:x2]
                generated.append((filename, mask_crop, box))
        return generated

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item: int) -> Dict:
        filename, mask_crop, box = self.samples[item]
        image = read_tiff_crop(filename, box)
        sample = {"img": image.astype(np.float32), "seg": mask_crop.astype(np.float32)}
        return sample if self.transform is None else self.transform(sample)


def create_data_loaders(
    reader: DatasetReader,
    transformers: Transformers,
    sliding_window_generator: SlidingWindowsGenerator,
    train_keys: Optional[List[str]] = None,
    num_workers: int = 0,
    batch_size: int = 4,
    outliers_threshold: Optional[int] = None
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
        loaders[subset] = DataLoader(
            dataset=OnlineCroppingDataset(
                keys=keys,
                reader=reader,
                sliding_windows_generator=sliding_window_generator,
                outliers_excluded=outliers_threshold is not None,
                outliers_threshold=outliers_threshold,
                transform=getattr(transformers, subset, None)
            ),
            batch_size=batch_size,
            shuffle=subset == "train",
            num_workers=num_workers,
            collate_fn=list_data_collate
        )
    return loaders


def main():
    reader = get_reader()
    keys = reader.get_keys(SampleType.Labeled)
    dataset = OnlineCroppingDataset(
        [keys[0]], reader,
        sliding_windows_generator=SlidingWindowsGenerator(1024, 32),
        outliers_threshold=10_000
    )
    x = dataset[0]
    print(x)


if __name__ == '__main__':
    main()
