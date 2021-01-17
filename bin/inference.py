import os
import sys
from argparse import Namespace, ArgumentParser
from typing import Tuple, Optional, Callable, Dict, List

import PIL.Image
import pandas as pd
import pytorch_lightning as pl
import torch
from prompt_toolkit import prompt

from kidney.datasets.kaggle import get_reader, SampleType
from kidney.inference.inference import SlidingWindow, SlidingWindowConfig
from kidney.utils.checkpoints import CheckpointsStorage, get_factory, load_experiment
from kidney.utils.mask import rle_numba_encode


def main(args: Namespace):
    experiment, meta = load_experiment(
        get_factory(args.factory_class),
        *checkpoint_picker(args.checkpoints_dir)
    )
    inference = inference_picker(
        experiment,
        device=args.device,
        debug=args.debug,
        transform_input=meta["transformers"].test_preprocessing,
        transform_output=meta["transformers"].test_postprocessing
    )
    encoder = (
        rle_numba_encode
        if prompt("decode masks? ", default="yes").strip().lower() == "yes"
        else None
    )
    predictions = inference.predict_from_reader(
        reader=get_reader(),
        sample_type=args.sample_type,
        encoder=encoder
    )
    save_results(predictions, args.output_file, encoder is not None)


def checkpoint_picker(checkpoints_dir: str) -> Tuple[str, str]:
    """Starts an interactive prompt that allows to choose an experiment
    factory and checkpoint to restore.
    """
    storage = CheckpointsStorage(checkpoints_dir)
    metric = prompt("performance metric: ", default="avg_val_loss")
    checkpoints = storage.fetch_available_checkpoints(metric, best_checkpoint_per_date=False)

    def pick_timestamp() -> str:
        print("available timestamps (non-empty):")
        counter = 0
        non_empty = []
        for name in checkpoints.names:
            result = checkpoints[name]
            filenames = result["checkpoints"]
            if not filenames:
                continue
            counter += 1
            non_empty.append(name)
            print(f"[{counter}] ts={name} with {len(filenames)} checkpoint(s)")
            for i, fn in enumerate(filenames, 1):
                print(f"\t{i})", os.path.basename(fn))
        index = prompt("pick timestamp number: ", default=str(len(non_empty)))
        return non_empty[int(index) - 1]

    def pick_checkpoint(timestamp: str) -> Tuple[str, str]:
        result = checkpoints[timestamp]
        filenames = result["checkpoints"]
        print("available checkpoints:")
        for i, filename in enumerate(filenames, 1):
            print(f"{i})", os.path.basename(filename))
        index = prompt("pick checkpoint: ", default=str(len(filenames)))
        return filenames[int(index) - 1], result["meta"]

    return pick_checkpoint(pick_timestamp())


def inference_picker(
    experiment: pl.LightningModule,
    device: torch.device,
    debug: bool = False,
    transform_input: Optional[Callable] = None,
    transform_output: Optional[Callable] = None,
):
    inference = SlidingWindow(
        model=experiment,
        device=device,
        debug=debug,
        config=SlidingWindowConfig(
            window_size=int(prompt("window size: ", default=str(512))),
            overlap=int(prompt("overlap: ", default=str(32))),
            max_batch_size=int(prompt("max batch size: ", default=str(64))),
            check_for_outliers=prompt("check for outliers: ", default="yes") == "yes",
            outliers_threshold=int(prompt("outliers threshold: ", default=str(1000))),
            transform_input=transform_input,
            transform_output=transform_output
        )
    )
    return inference


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
    parser.add_argument("--factory_class", required=True)
    parser.add_argument("--checkpoints_dir", required=True)
    parser.add_argument('--sample_type',
                        choices=[case.name for case in SampleType],
                        default=SampleType.Unlabeled.name)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_file", default="/tmp/predictions.csv")
    args = parser.parse_args()
    args.sample_type = SampleType[args.sample_type]
    return args


if __name__ == '__main__':
    try:
        main(parse_args())
    except KeyboardInterrupt:
        sys.exit(1)
