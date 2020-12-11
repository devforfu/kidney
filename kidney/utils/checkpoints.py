from dataclasses import dataclass
from os.path import exists, join
from typing import Type, Dict, Tuple, Optional

import torch
import pytorch_lightning as pl
from zeus.torch_tools.checkpoints import find_latest_dir, find_best_file
from zeus.utils import TimestampFormat


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

    def fetch_best_file(self, metric: str, with_meta: bool = True) -> Checkpoint:
        latest_dir = find_latest_dir(self.folder, str(self.timestamp_format))
        meta_file = join(latest_dir, self.meta_name)

        meta = None
        if not exists(meta_file) and with_meta:
            raise FileNotFoundError(meta_file)
        elif exists(meta_file):
            meta = torch.load(meta_file)

        best_file = find_best_file(latest_dir, metric=metric, extension=self.extension)
        return Checkpoint(best_file, meta)

    def restore_experiment(
        self,
        factory: Type[pl.LightningModule],
        metric: str,
        **restore_kwargs: Dict
    ) -> Tuple[pl.LightningModule, Checkpoint]:
        checkpoint = self.fetch_best_file(metric, with_meta=True)
        experiment = factory.load_from_checkpoint(
            checkpoint.path,
            **restore_kwargs,
            params=checkpoint.meta["params"]
        )
        return experiment, checkpoint
