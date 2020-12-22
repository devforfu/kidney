import importlib
from argparse import Namespace, ArgumentParser
from multiprocessing import cpu_count

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities import AttributeDict

from kidney.datasets.kaggle import get_reader, SampleType
from kidney.inference.inference import create_inference_instance, InferenceAlgorithm
from kidney.parameters import as_attribute_dict, requires
from kidney.utils.checkpoints import CheckpointsStorage
from kidney.utils.mask import rle_encode


def main(args: Namespace):
    inference(as_attribute_dict(args))


@requires([
    "experiment",
    "checkpoints",
    "metric",
    "data_loaders",
    "num_workers",
    "batch_size",
    "subset",
    "validation",
    "roi_size",
    "batch_size"
])
def inference(args: AttributeDict):
    experiment_factory = get_module_attribute(args.experiment)
    experiment, checkpoint = CheckpointsStorage.create(experiment_factory, args.checkpoints, args.metric)
    create_data_loaders = get_module_attribute(args.data_loaders)

    meta = checkpoint.meta
    transformers = meta["transformers"]
    loaders = create_data_loaders(meta["data"], transformers,
                                  num_workers=args.num_workers,
                                  batch_size=args.batch_size)

    if args.validation:
        trainer = pl.Trainer(gpus=1)
        trainer.test(experiment, test_dataloaders=loaders[args.subset])
    else:
        predictor = create_inference_instance(
            algorithm=InferenceAlgorithm.Default,
            experiment=experiment,
            device=args.device,
            transformers=transformers,
            check_for_outliers=False
        )
        reader = get_reader()
        test_keys = reader.get_keys(SampleType.Unlabeled)
        predictions = []
        for key in test_keys:
            meta = reader.fetch_meta(key)
            predicted = predictor.predict_from_file(meta["tiff"])
            encoded = rle_encode(predicted)
            predictions.append((key, encoded))
        predictions = pd.DataFrame(columns=["id", "predicted"], data=predictions)
        predictions.to_csv(args.output_file, index=False)


def get_module_attribute(import_path: str):
    module_path, attr = import_path.rsplit(".", 1)
    assert module_path and attr
    module = importlib.import_module(module_path)
    try:
        return getattr(module, attr)
    except AttributeError:
        raise RuntimeError(f"module is not found: {import_path}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-e", "--experiment", required=True)
    parser.add_argument("-d", "--data_loaders", required=True)
    parser.add_argument("-c", "--checkpoints", required=True)
    parser.add_argument("-s", "--subset", default="valid")
    parser.add_argument("-m", "--metric", default="avg_val_loss")
    parser.add_argument("-w", "--num_workers", default=cpu_count(), type=int)
    parser.add_argument("-b", "--batch_size", default=4, type=int)
    parser.add_argument("-r", "--roi_size", default=1024, type=int)
    parser.add_argument("-dev", "--device", default="cuda:1")
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("-o", "--output_file", default="submission.csv")
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
