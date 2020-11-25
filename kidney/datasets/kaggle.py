import glob
from collections import defaultdict
from dataclasses import dataclass
from enum import auto, Enum
from os.path import basename, splitext
from typing import List, Any, Dict, Optional

import pandas as pd

from kidney.utils.mask import rle_decode
from kidney.utils.tiff import read_tiff


class SampleType(Enum):
    Labeled = auto()
    Unlabeled = auto()
    All = auto()


class DatasetReader:

    def get_keys(self, sample_type: SampleType = SampleType.All) -> List[str]:
        """Returns a list of keys available for a given sample type."""

    def fetch_one(self, key: str) -> Dict[str, Any]:
        """Fetches dataset's sample and target (if available) for a given key."""

    def fetch_meta(self, key: str) -> Dict[str, Any]:
        """Fetches meta-information about sample for a given key."""


@dataclass
class KaggleKidneyDatasetReader(DatasetReader):
    """Kaggle Kidney dataset information reader."""

    root: str
    samples: Optional[Dict[str, Dict]] = None
    masks: Optional[Dict[str, str]] = None

    def __post_init__(self):
        samples = defaultdict(dict)
        for subset in ('train', 'test'):
            for fn in sorted(glob.glob(f'{self.root}/{subset}/*')):
                name, ext = splitext(fn)
                identifier = basename(name)
                if ext == '.tiff':
                    samples[identifier]['tiff'] = fn
                elif ext == '.json':
                    if 'anatomical-structure' in name:
                        identifier, *_ = identifier.split('-')
                        samples[identifier]['anatomical_structure'] = fn
                    else:
                        samples[identifier]['target_mask'] = fn
                else:
                    continue
                samples[identifier]['subset'] = subset
        masks = pd.read_csv(f'{self.root}/train.csv').set_index('id')
        self.masks = masks.encoding.to_dict()
        self.samples = dict(samples)

    def get_keys(self, sample_type: SampleType = SampleType.All) -> List[str]:
        include = (
            lambda sample:
            sample['subset'] == 'train' if sample_type == SampleType.Labeled else
            sample['subset'] == 'test' if sample_type == SampleType.Unlabeled else
            True
        )
        keys = [
            key
            for key, sample in self.samples.items()
            if include(sample)
        ]
        return keys

    def fetch_meta(self, key: str) -> Dict[str, Any]:
        assert key in self.samples, 'Key is not found'
        mask = self.masks.get(key)
        sample = self.samples[key].copy()
        sample['mask'] = mask
        return sample

    def fetch_one(self, key: str) -> Dict[str, Any]:
        meta = self.fetch_meta(key)
        image = read_tiff(meta['tiff'])
        mask = rle_decode(meta['mask'], image.shape[1:])
        return {'image': image, 'mask': mask}


def main():
    reader = KaggleKidneyDatasetReader('/mnt/fast/data/kidney')
    print(reader.fetch_one('0486052bb').keys())


if __name__ == '__main__':
    main()
