import os
import sys
from argparse import Namespace, ArgumentParser
from typing import List, Dict

import pandas as pd
import PIL.Image
import torch
from prompt_toolkit import prompt

from kidney.datasets.kaggle import SampleType, get_reader
from kidney.inference.inference import SlidingWindow, SlidingWindowConfig
from kidney.utils.checkpoints import load_experiment, get_factory
from kidney.utils.mask import rle_numba_encode


def main(args: Namespace):
    experiment, meta = load_experiment(
        get_factory(prompt("factory class: ", default="kidney.experiments.smp.SMPExperiment")),
        checkpoint_file=args.checkpoint_file,
        meta_file=args.meta_file
    )
    inference = SlidingWindow(
        model=experiment,
        device=torch.device(prompt("device: ", default="cuda:1")),
        debug=prompt("debug? ", default="no").lower() == "yes",
        config=SlidingWindowConfig(
            window_size=int(prompt("window size: ", default="1024")),
            overlap=int(prompt("overlap: ", default="32")),
            max_batch_size=int(prompt("max batch size: ", default="64")),
            check_for_outliers=prompt("check for outliers: ", default="yes").lower() == "yes",
            outliers_threshold=int(prompt("outliers threshold: ", default="1000")),
            transform_input=meta["transformers"].test_preprocessing,
            transform_output=meta["transformers"].test_postprocessing
        )
    )
    encoder = (
        rle_numba_encode
        if prompt("encode masks? ", default="yes").strip().lower() == "yes"
        else None
    )
    output_file = prompt("output file: ")
    predictions = inference.predict_from_reader(
        reader=get_reader(),
        sample_type=SampleType[prompt("sample type: ", default=SampleType.All.name)],
        encoder=encoder
    )
    save_results(predictions, output_file, encoder is not None)


def save_results(predictions: List[Dict], output_file: str, encoded: bool):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if encoded:
        predictions_df = pd.DataFrame(predictions, columns=["id", "predicted"])
        predictions_df.to_csv(output_file, index=False)
        print("Predictions path:", output_file)
    else:
        os.makedirs(output_file, exist_ok=True)
        for result in predictions:
            image_id = result["id"]
            mask = result["predicted"]
            if mask.max() == 1:
                mask *= 255
            image_file = os.path.join(output_file, f"{image_id}.png")
            PIL.Image.fromarray(mask).save(image_file, format="png")
            print("Mask saved:", image_file)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-c", dest="checkpoint_file", required=True)
    parser.add_argument("-m", dest="meta_file", required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    try:
        main(parse_args())
    except KeyboardInterrupt:
        sys.exit(1)
