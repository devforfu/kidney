import importlib
from argparse import Namespace, ArgumentParser
from multiprocessing import cpu_count

import pytorch_lightning as pl
from zeus.utils import format_dict

from kidney.utils.checkpoints import CheckpointsStorage


def main(args: Namespace):
    experiment_factory = get_module_attribute(args.experiment)
    experiment, checkpoint = CheckpointsStorage.create(experiment_factory, args.checkpoints, args.metric)
    create_data_loaders = get_module_attribute(args.data_loaders)

    meta = checkpoint.meta
    loaders = create_data_loaders(meta["data"], meta["transformers"],
                                  num_workers=args.num_workers,
                                  batch_size=args.batch_size)

    trainer = pl.Trainer(gpus=1)
    trainer.test(experiment, test_dataloaders=loaders["valid"])


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
    parser.add_argument("-dev", "--device", default="cuda:1")
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
