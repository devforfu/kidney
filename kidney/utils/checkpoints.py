import importlib
import os
from collections import OrderedDict
from dataclasses import dataclass
from operator import itemgetter
from os.path import exists, join
from typing import Type, Dict, Tuple, Optional

import pytorch_lightning as pl
import torch
from zeus.torch_tools.checkpoints import find_latest_dir, find_best_file
from zeus.utils import TimestampFormat, list_files, NamedList


@dataclass
class Checkpoint:
    path: str
    meta: Optional[Dict] = None


@dataclass
class CheckpointsStorage:
    folder: str
    extension: str = ".ckpt"
    meta_name: str = "info.pth"
    timestamp_format: TimestampFormat = TimestampFormat.VerboseShortSeconds

    @classmethod
    def create(cls, factory: Type[pl.LightningModule], folder: str, metric: str):
        storage = CheckpointsStorage(folder)
        model, checkpoint = storage.restore_experiment(factory, metric)
        return model, checkpoint

    def __post_init__(self):
        assert exists(self.folder)

    def fetch_available_checkpoints(
        self,
        metric: str,
        best_checkpoint_per_date: bool = True,
        with_meta: bool = True
    ) -> NamedList:
        timestamps = sorted(
            [
                (join(self.folder, t), self.timestamp_format.parse(t))
                for t in os.listdir(self.folder)
            ],
            key=itemgetter(1)
        )
        checkpoints = OrderedDict()

        for ts, _ in timestamps:
            key = os.path.basename(ts)

            if with_meta:
                meta_file = join(ts, self.meta_name)
                if not os.path.exists(meta_file):
                    continue
            else:
                meta_file = None

            if best_checkpoint_per_date:
                filename = find_best_file(ts, metric=metric, extension=self.extension)
                checkpoints[key] = {"checkpoint": filename, "meta": meta_file}

            else:
                checkpoints_per_subdir = []
                for filename in list_files(ts):
                    if filename.endswith(self.meta_name):
                        continue
                    checkpoints_per_subdir.append(filename)
                checkpoints[key] = {"checkpoints": checkpoints_per_subdir, "meta": meta_file}

        return NamedList(checkpoints)

    def fetch_best_file(self, metric: str, with_meta: bool = True) -> Checkpoint:
        """Fetches the best (according to given `metric`) checkpoint from the
        latest checkpoint's folder.

        Each checkpoint is expected to follow a certain naming condition that includes
        checkpoint's performance metrics, like:

            epoch=<number>_val_loss=<number>.<extension>

        And each checkpoint's folder should have a certain structure, like:

            /checkpoints
                /Mon_02_Mar__04_05_06
                    /epoch=<epoch_1>_val_loss=<loss_1>.<extension>
                    /epoch=<epoch_2>_val_loss=<loss_2>.<extension>
                    ...
                    /<meta_name>
                /Tue_03_Apr__05_06_07
                    ...

        Only the folder with the latest timestamp is considered, and the best
        checkpoint is discovered in this folder.

        Parameters
        ----------
        metric
            Metric used to determine the best checkpoint file. Should be a part of
            checkpoints' filenames.
        with_meta
            If True, each checkpoint is expected to have an accompanying file with
            meta-information storing experiment's parameters and configurations.

        Returns
        -------
        Checkpoint
            The structure with best file paths.

        """
        latest_dir = find_latest_dir(self.folder, str(self.timestamp_format))
        return self.fetch_best_file_in_dir(latest_dir, metric, with_meta)

    def fetch_best_file_in_dir(self, metric: str, with_meta: bool = True, dirname: Optional[str] = None) -> Checkpoint:
        dirname = dirname or self.folder
        meta_file = join(dirname, self.meta_name)

        meta = None
        if not exists(meta_file) and with_meta:
            raise FileNotFoundError(meta_file)
        elif exists(meta_file):
            meta = torch.load(meta_file)

        best_file = find_best_file(dirname, metric=metric, extension=self.extension)
        return Checkpoint(best_file, meta)


    def restore_experiment(
        self,
        factory: Type[pl.LightningModule],
        metric: str,
        **restore_kwargs: Dict
    ) -> Tuple[pl.LightningModule, Checkpoint]:
        """Restores the best checkpoint for a given experiment.

        Parameters
        ----------
        factory
            Experiment's factory.
        metric
            Metric used to determine the best checkpoint file.
        restore_kwargs
            Additional parameters given to `factory.load_from_checkpoint` method.

        Returns
        -------
        Tuple
            The restored experiment and information about checkpoint.

        """
        checkpoint = self.fetch_best_file(metric, with_meta=True)
        experiment = factory.load_from_checkpoint(
            checkpoint.path,
            **restore_kwargs,
            params=checkpoint.meta["params"]
        )
        return experiment, checkpoint


def load_experiment(factory, checkpoint_file, meta_file, strict=False):
    if isinstance(meta_file, str):
        meta = torch.load(meta_file)
    else:
        meta = meta_file
    experiment = factory.load_from_checkpoint(
        checkpoint_file, params=meta["params"], strict=strict)
    experiment = experiment.eval()
    return experiment, meta


def load_experiment_from_dict(factory, checkpoint_dict: Dict[str, str], **load_opts):
    return load_experiment(factory, checkpoint_dict["checkpoint"], checkpoint_dict["meta"], **load_opts)


def get_factory(import_path: str) -> Type[pl.LightningModule]:
    try:
        module_path, class_name = import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ValueError, AttributeError):
        raise ImportError(f"cannot import class: {import_path}")
