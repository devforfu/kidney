import argparse
import os
import uuid
from functools import partial
from logging import basicConfig
from typing import Optional, List, Dict

import dask.bag as db
import srsly
from distributed import Client

from kidney.datasets.kaggle import get_reader, SampleType, DatasetReader, outlier
from kidney.datasets.online import read_boxes
from kidney.inference.window import SlidingWindowsGenerator
from kidney.log import get_logger
# from kidney.utils import rle
from kidney.utils import rle
from kidney.utils.mask import rle_decode, rle_numba_encode
from kidney.utils.tiff import read_tiff, read_tiff_crop

basicConfig()


def main(args: argparse.Namespace):
    reader = get_reader()
    generator = SlidingWindowsGenerator(args.window_size, args.overlap)
    train_keys = reader.get_keys(SampleType.Labeled)

    client = Client()
    try:
        client.run(lambda: basicConfig())
        bag = (
            db.from_sequence(train_keys)
            .map(partial(generate_boxes,
                         reader=reader,
                         generator=generator,
                         histogram_threshold=args.histogram_threshold,
                         mask_threshold=args.mask_threshold))
            .map(partial(save_outputs, output_dir=args.output_dir))
        )
        filenames = bag.compute(scheduler=args.dask_scheduler)
    finally:
        client.close()

    logger = get_logger(__name__)
    for filename in filenames:
        logger.info(filename)

    logger.info(f"Total number of saved files: {len(filenames)}")
    logger.info(f"Total number of boxes: {len(read_boxes(args.output_dir))}")


def full_column(arr):
    col_sums = arr.sum(axis=0)
    height = arr.shape[0]
    return (col_sums == height).any()


def full_row(arr):
    row_sums = arr.sum(axis=1)
    width = arr.shape[1]
    return (row_sums == width).any()


def generate_boxes(
    key: str,
    generator: SlidingWindowsGenerator,
    reader: DatasetReader,
    histogram_threshold: Optional[int] = None,
    mask_threshold: Optional[int] = None
):
    logger = get_logger(__name__)

    meta = reader.fetch_meta(key)
    filename = meta["tiff"]
    logger.info(f"reading TIFF file: {filename}")

    boxes, (h, w) = generator.generate(filename)
    mask = rle.decode(meta["mask"], shape=(h, w))

    generated = []
    for box in boxes:
        x1, y1, x2, y2 = [x.item() for x in box]
        mask_crop = mask[y1:y2, x1:x2]
        if mask_threshold is not None:
            if mask_crop.sum() <= mask_threshold:
                continue
        if histogram_threshold is not None:
            crop = read_tiff_crop(filename, box)
            if outlier(crop, threshold=histogram_threshold):
                continue
        encoded = rle.encode(mask_crop)
        record = {"key": key,
                  "rle_encoded": encoded,
                  "width": x2 - x1,
                  "height": y2 - y1,
                  "box": [x1, y1, x2, y2]}
        logger.info(f"{key}: box={box.tolist()}")
        generated.append(record)
    return generated


def save_outputs(records: List[Dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{uuid.uuid4()}.jsonl")
    srsly.write_jsonl(filename, records)
    get_logger(__name__).info(f"saved {len(records)} boxes: {filename}")
    return filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--histogram-threshold", type=int, default=None)
    parser.add_argument("--mask-threshold", type=int, default=None)
    parser.add_argument("--dask-scheduler", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
