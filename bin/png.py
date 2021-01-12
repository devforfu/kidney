import os
from argparse import Namespace, ArgumentParser
from functools import partial
from logging import basicConfig
from typing import Dict, List

import numpy as np
import dask.bag as db
import PIL.Image
from distributed import Client

from kidney.datasets.kaggle import SampleType, KaggleKidneyDatasetReader, DatasetReader
from kidney.datasets.utils import read_boxes
from kidney.log import get_logger

basicConfig()


def main(args: Namespace):
    logger = get_logger(__name__)
    logger.info("reading samples from dir: %s", args.images_dir)
    reader = KaggleKidneyDatasetReader(args.images_dir)

    keys = reader.get_keys(SampleType.All)
    logger.info("retrieved keys: %s", ", ".join(keys))

    logger.info("reading boxes from dir: %s", args.boxes_dir)
    boxes = read_boxes(args.boxes_dir)
    logger.info("the number of boxes: %d", len(boxes))

    logger.info("creating output dir: %s", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    client = Client()
    try:
        client.run(lambda: basicConfig())
        bag = (
            db.from_sequence(keys, npartitions=4)
            .map(partial(read_from_disk, reader=reader, boxes=boxes))
            .map(partial(cut_image, output_dir=args.output_dir))
        )
        logger.info("running dask pipeline")
        bag.compute()
    finally:
        logger.info("closing dask client")
        client.close()


def read_from_disk(key: str, reader: DatasetReader, boxes: List[Dict]) -> Dict:
    logger = get_logger(__file__)
    logger.info("fetching key: %s", key)
    sample = reader.fetch_one(key)
    sample["key"] = key
    sample["boxes"] = [box for box in boxes if box["key"] == key]
    return sample


def cut_image(sample: Dict, output_dir: str):
    logger = get_logger(__file__)
    key, img, seg = sample["key"], sample["image"], sample.get("mask")

    for bbox in sample["boxes"]:
        x1, y1, x2, y2 = bbox["box"]

        crop_img = img[y1:y2, x1:x2]
        fn_img = os.path.join(output_dir, f"img.{key}_{x1}_{y1}_{x2}_{y2}.png")
        logger.info("saving image: %s", fn_img)
        PIL.Image.fromarray(crop_img).save(fn_img)

        if seg is not None:
            crop_seg = (seg[y1:y2, x1:x2] * 255).astype(np.uint8)
            fn_seg = os.path.join(output_dir, f"seg.{key}_{x1}_{y1}_{x2}_{y2}.png")
            logger.info("saving mask: %s", fn_seg)
            PIL.Image.fromarray(crop_seg).save(fn_seg)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--boxes_dir", required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
