from typing import Dict

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.utilities import AttributeDict
from zeus.core.random import super_seed

from kidney.cli import entry_point, default_args
from kidney.cli.basic import basic_parser
from kidney.cli.lightning import make_trainer_init_params
from kidney.cli.models import add_fcn_args
from kidney.datasets.kaggle import get_reader
from kidney.datasets.online import create_data_loaders
from kidney.datasets.transformers import create_monai_crop_to_many_sigmoid_transformers, IntensityNormalization
from kidney.experiments import BaseExperiment
from kidney.inference.window import SlidingWindowsGenerator
from kidney.log import get_logger
from kidney.models.fcn import create_fcn_model


@entry_point(
    base_parser_factory=lambda: basic_parser(__file__),
    extensions=default_args() + [add_fcn_args]
)
def main(params: AttributeDict):
    super_seed(params.seed)

    logger = get_logger(__file__)

    reader = get_reader()
    transforms = create_monai_crop_to_many_sigmoid_transformers(
        image_key="img",
        mask_key="seg",
        image_size=1024,
        crop_fraction=0.5,
        crop_balanced=False,
        load_from_disk=False,
        as_channels_first=False,
        normalization=IntensityNormalization.ImageNet
    )
    loaders = create_data_loaders(
        reader=reader,
        transformers=transforms,
        sliding_window_generator=SlidingWindowsGenerator(
            window_size=1024,
            overlap=32,
            limit=10 if params.dataset == "debug" else None
        ),
        num_workers=params.num_workers,
        batch_size=params.batch_size,
        outliers_threshold=10000
    )

    trainer = pl.Trainer(**make_trainer_init_params(params))

    trainer.fit(model=FCNExperiment(params),
                train_dataloader=loaders["train"],
                val_dataloaders=loaders["valid"])


class FCNExperiment(BaseExperiment):

    def create_model(self) -> nn.Module:
        return create_fcn_model(self.hparams)

    def forward(self, batch: Dict) -> Dict:
        outputs = self.model(batch["img"])
        predicted_mask = outputs["out"]
        if "seg" in batch:
            loss = self.loss_fn(predicted_mask, batch["seg"])
            return {"loss": loss, "outputs": predicted_mask}
        return {"outputs": predicted_mask}


if __name__ == '__main__':
    main()
