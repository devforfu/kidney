"""Reads TIFF images and cuts them into smaller pieces ready for model training."""
import os
from argparse import ArgumentParser, Namespace
from functools import partial
from logging import getLogger, basicConfig
from os.path import join
from typing import Dict

import PIL.Image
import dask.bag as db
from distributed import Client

from kidney.datasets.kaggle import KaggleKidneyDatasetReader, SampleType, DatasetReader
from kidney.log import get_logger


basicConfig()


def main(args: Namespace):
    logger = get_logger(__name__)
    logger.info('reading samples from dir: %s', args.input_dir)
    reader = KaggleKidneyDatasetReader(args.input_dir)

    logger.info('sample type: %s', args.sample_type)
    keys = reader.get_keys(args.sample_type)
    logger.info('retrieved keys: %s', ', '.join(keys))

    client = Client()
    try:
        client.run(lambda: basicConfig())
        bag = (
            db.from_sequence(keys, npartitions=4)
            .map(partial(read_from_disk, reader=reader))
            .map(partial(
                generate_patches,
                size=args.patch_size,
                stride=args.patch_stride,
                output_dir=args.output_dir
            ))
        )
        logger.info('running dask pipeline')
        bag.compute()
    finally:
        logger.info('closing dask client')
        client.close()


def read_from_disk(key: str, reader: DatasetReader) -> Dict:
    sample = reader.fetch_one(key)
    sample['key'] = key
    return sample


def generate_patches(
    sample: Dict,
    size: int,
    stride: int,
    output_dir: str,
    drop_last: bool = True
) -> None:
    logger = get_logger(__name__)

    os.makedirs(output_dir, exist_ok=True)

    image = sample['image']
    key = sample['key']
    mask = sample.get('mask')

    if mask is not None:
        if (
            image.shape[0] != mask.shape[0] or
            image.shape[1] != mask.shape[1]
        ):
            logger.error(f'key {key} failed: image and mask shapes are not equal')
            return

        if mask.ndim != 2:
            logger.error(f'key {key} failed: mask should be binary')
            return

    y_size, x_size = image.shape[:2]

    if drop_last:
        # otherwise, the last patches can be smaller than (size, size) shape;
        # ignoring them for now but consider padding edge patches instead
        y_size -= size
        x_size -= size

    for dy in range(0, y_size, stride):
        for dx in range(0, x_size, stride):

            path = join(output_dir, f'img.{key}.{dx}.{dy}.{size}.png')
            logger.info('saving image: %s', path)
            patch_image = PIL.Image.fromarray(image[dy:dy + size, dx:dx + size])
            patch_image.save(path, format='png')

            if mask is not None:
                path = join(output_dir, f'seg.{key}.{dx}.{dy}.{size}.png')
                logger.info('saving mask: %s', path)
                patch_mask = PIL.Image.fromarray(mask[dy:dy + size, dx:dx + size])
                patch_mask.save(path, format='png')


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--patch-size', type=int, default=512)
    parser.add_argument('--patch-stride', type=int, default=512)
    parser.add_argument('--drop-last', action='store_true')
    parser.add_argument('--sample-type',
                        choices=[case.name for case in SampleType],
                        default=SampleType.Labeled.name)
    args = parser.parse_args()
    args.sample_type = SampleType[args.sample_type]
    return args


if __name__ == '__main__':
    main(parse_args())
