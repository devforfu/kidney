"""Reads TIFF images and cuts them into smaller pieces ready for model training."""

import os
from argparse import ArgumentParser, Namespace
from functools import partial
from logging import getLogger
from os.path import join
from typing import Dict

import PIL.Image
import dask.bag as db
from distributed import Client

from kidney.datasets.kaggle import KaggleKidneyDatasetReader, SampleType, DatasetReader

_logger = getLogger(__name__)


def main(args: Namespace):
    _logger.info('reading samples from dir: %s', args.input_dir)
    reader = KaggleKidneyDatasetReader(args.input_dir)

    _logger.info('sample type: %s', args.sample_type)
    keys = reader.get_keys(args.sample_type)
    _logger.info('retrieved keys: %s', ', '.join(keys))

    client = Client()
    try:
        bag = (
            db.from_sequence(keys, npartitions=len(keys))
            .map(partial(read_from_disk, reader=reader))
            .map(partial(generate_patches, size=args.patch_stride, stride=args.patch_size, output_dir=args.output_dir))
        )
        bag.compute(scheduler='synchronous')
    finally:
        client.close()


def read_from_disk(key: str, reader: DatasetReader) -> Dict:
    sample = reader.fetch_one(key)
    sample['key'] = key
    return sample


def generate_patches(sample: Dict, size: int, stride: int, output_dir: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    image = sample['image']
    key = sample['key']
    mask = sample.get('mask')

    if mask is not None:
        if (
            image.shape[0] != mask.shape[0] or
            image.shape[1] != mask.shape[1]
        ):
            return {'key': key, 'error': 'image and mask shape do not match'}

        if mask.ndim != 2:
            return {'key': key, 'error': 'mask is not binary'}

    y_size, x_size = image.shape[:2]

    # todo: consider padding edge patches instead of ignoring them
    for dy in range(0, y_size, stride):
        for dx in range(0, x_size, stride):

            path = join(output_dir, f'img.{dx}.{dy}.{size}.png')
            image = PIL.Image.fromarray(image[dy:dy + size, dx:dx + size])
            image.save(path, format='png')

            if mask is not None:
                path = join(output_dir, f'seg.{dx}.{dy}.{size}.png')
                image = PIL.Image.fromarray(mask[dy:dy + size, dx:dx + size])
                image.save(path, format='png')


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--patch-size', default=512)
    parser.add_argument('--patch-stride', default=512)
    parser.add_argument('--drop-last', action='store_true')
    parser.add_argument('--sample-type',
                        choices=[case.name for case in SampleType],
                        default=SampleType.Labeled.name)
    args = parser.parse_args()
    args.sample_type = SampleType[args.sample_type]
    return args


if __name__ == '__main__':
    main(parse_args())
