import logging
import os
from argparse import Namespace, ArgumentParser
from logging import basicConfig
from os.path import join

from kidney.cli import parse_callable_definition
from kidney.datasets.kaggle import get_reader, SampleType
from kidney.tools.cutter import cut_sample, NoOpFilter, HistogramFilter, SaturationFilter, MaskFilter

CONFIG = {
    "256px": {
        "reduce": 4,
        "tile_size": 256,
    },
    "1024px": {
        "reduce": 1,
        "tile_size": 1024
    }
}

basicConfig()
log = logging.getLogger()
log.setLevel(logging.INFO)


def main(args: Namespace):
    log.info("configuration: %s", args.config)
    reduce_multiplier = args.config["reduce"]
    tile_size = args.config["tile_size"]
    reader = get_reader()

    log.info("writing images into folder: %s", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("processing samples of type: %s", args.sample_type.name)
    log.info("using outliers filter: %s", args.outliers_filter)

    for key in reader.get_keys(args.sample_type):
        log.info("processing key: %s", key)
        sample = reader.fetch_one(key)

        crops = cut_sample(sample, tile_size, args.outliers_filter, reduce_multiplier)
        log.info("number of generated crops: %d", len(crops))

        for i, crop in enumerate(crops):
            img_path = join(args.output_dir, f"img.{key}_{i}.png")
            log.info("saving image: %s", img_path)
            crop["img"].save(img_path)

            if "seg" in crop:
                seg_path = join(args.output_dir, f"seg.{key}_{i}.png")
                log.info("saving mask: %s", seg_path)
                crop["seg"].save(seg_path)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIG), default="1024px")
    parser.add_argument("--output-dir", default="/tmp/images")
    parser.add_argument("--sample-type", default=SampleType.Labeled.name)
    parser.add_argument("--outliers-filter", type=create_outliers_filter, default="hist")
    args = parser.parse_args()
    args.config = CONFIG[args.config]
    args.sample_type = SampleType[args.sample_type]
    return args


def create_outliers_filter(definition: str):
    name, params = parse_callable_definition(definition)
    try:
        factory = {
            "noop": NoOpFilter,
            "hist": HistogramFilter,
            "hsv": SaturationFilter,
            "mask": MaskFilter,
        }[name]
    except KeyError:
        raise NotImplementedError(f"unknown filtering function: {name}")
    return factory(**params)


if __name__ == '__main__':
    main(parse_args())
