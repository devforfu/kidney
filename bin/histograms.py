import argparse
import json
from functools import partial
from logging import getLogger, basicConfig
from multiprocessing import cpu_count
from typing import Dict, Any

import dask.bag as db
from distributed import Client

from kidney.datasets.segmentation import read_masked_images
from kidney.log import get_logger
from kidney.utils.image import read_image_as_numpy, pixel_histogram


basicConfig()


def main(args: argparse.Namespace):
    logger = getLogger(__name__)
    logger.info('building images meta-info from folder: %s', args.input_dir)
    images = read_masked_images(args.input_dir)
    logger.info('the number of discovered samples: %d', len(images))

    client = Client()
    try:
        client.run(lambda: basicConfig())
        bag = (
            db.from_sequence(images, npartitions=args.n_partitions)
            .map(partial(compute_histogram, bin_size=args.bin_size))
            .map(partial(classify_sample, threshold=args.threshold))
        )
        results = bag.compute()
    finally:
        logger.info('closing dask client')
        client.close()

    with open(args.output_file, 'w') as fp:
        json.dump(results, fp, indent=2)

    logger.info('histograms saved: %s', args.output_file)


def compute_histogram(sample: Dict[str, Any], bin_size: int) -> Dict[str, Any]:
    logger = get_logger(__name__)
    path = sample['image']
    logger.info('computing histogram: %s', path)
    image = read_image_as_numpy(path)
    sample['hist'] = pixel_histogram(image, bin_size)
    return sample


def classify_sample(sample: Dict[str, Any], threshold: int = 1000):
    logger = get_logger(__name__)
    logger.info('classifying image: %s', sample['image'])
    hist = sample['hist']
    df_hist = (
        hist.reset_index()
        .sort_values('count')
        .reset_index(drop=True)
    )
    median = len(hist) // 2
    count = df_hist.iloc[median]['count'].item()
    sample['relevant'] = count >= threshold
    sample['hist'] = hist.to_dict()
    return sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--n-partitions', type=int, default=cpu_count())
    parser.add_argument('--bin-size', type=int, default=4)
    parser.add_argument('--threshold', type=int, default=1000)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
