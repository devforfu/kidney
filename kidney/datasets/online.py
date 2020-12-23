from typing import List, Dict, Tuple

import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset

from kidney.datasets.kaggle import DatasetReader, SampleType, get_reader, outlier
from kidney.inference.window import SlidingWindowsGenerator
from kidney.log import get_logger
from kidney.utils.mask import rle_decode
from kidney.utils.tiff import read_tiff_crop

logger = get_logger(__file__)


class OnlineCroppingDataset(Dataset):

    def __init__(
        self,
        keys: List[str],
        reader: DatasetReader,
        sliding_windows_generator: SlidingWindowsGenerator,
        outliers_excluded: bool = True,
        outliers_threshold: int = 1000
    ):
        self.keys = keys
        self.reader = reader
        self.sliding_windows_generator = sliding_windows_generator
        self.samples = self._generate_samples(outliers_excluded, outliers_threshold)

    def _generate_samples(self, exclude_outliers: bool, threshold: int) -> List[Tuple]:
        generated = []
        for key in self.keys:
            logger.debug(f"reading key: {key}")
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
        return {"img": image, "seg": mask_crop}


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
