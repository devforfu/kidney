import os
from argparse import Namespace, ArgumentParser

import json
import pandas as pd

from kidney.datasets.kaggle import get_reader, SampleType
from kidney.inference.inference import SlidingWindow, SlidingWindowConfig
from kidney.utils.checkpoints import CheckpointsStorage, load_experiment, get_factory
from kidney.utils.mask import rle_numba_encode


def main(args: Namespace):
    reader = get_reader()

    checkpoints_dir = os.path.join(args.folds_dir, "checkpoints")

    print("Processing checkpoints from dir:", checkpoints_dir)
    storage = CheckpointsStorage(folder=checkpoints_dir)
    checkpoints = storage.fetch_available_checkpoints(metric="avg_val_dice_metric")
    total = len(checkpoints)
    os.makedirs(args.output_dir, exist_ok=True)

    for fold, files in enumerate(checkpoints, 1):
        print(f"Inference with fold {fold} of {total}")
        experiment, meta = load_experiment(
            factory=get_factory(args.factory_class),
            checkpoint_file=files["checkpoint"],
            meta_file=files["meta"]
        )
        inference = SlidingWindow(
            model=experiment,
            device=args.device,
            debug=args.debug,
            config=SlidingWindowConfig(
                **args.config,
                transform_input=meta["transformers"].test_preprocessing,
                transform_output=meta["transformers"].test_postprocessing
            )
        )
        predictions = inference.predict_from_reader(
            reader=reader,
            sample_type=args.sample_type,
            encoder=rle_numba_encode
        )
        filename = os.path.join(args.output_dir, f"fold_{fold}.csv")
        predictions_df = pd.DataFrame(predictions, columns=["id", "predicted"])
        predictions_df.to_csv(filename, index=False)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--folds_dir", required=True)
    parser.add_argument("--factory_class", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", type=json.loads, default="{}")
    parser.add_argument('--sample_type',
                        choices=[case.name for case in SampleType],
                        default=SampleType.Unlabeled.name)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    args.sample_type = SampleType[args.sample_type]
    return args


if __name__ == '__main__':
    main(parse_args())
