"""A toy experiment using randomly generated images and MONAI models."""
from multiprocessing import cpu_count
from typing import Dict

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.utilities import AttributeDict
from zeus.core.random import super_seed

from kidney.cli import entry_point, default_args
from kidney.cli.basic import basic_parser
from kidney.cli.lightning import make_trainer_init_params
from kidney.cli.models import add_unet_args
from kidney.datasets.toy import generate_synthetic_data, create_data_loaders, create_transformers
from kidney.experiments import BaseExperiment, save_experiment_info
from kidney.log import get_logger
from kidney.models.unet import create_unet_model


@entry_point(
    base_parser_factory=lambda: basic_parser(__file__),
    extensions=default_args() + [add_unet_args]
)
def main(params: AttributeDict):
    super_seed(params.seed)

    logger = get_logger(__name__)

    logger.info("generating synthetic data")
    data = generate_synthetic_data(1000)

    logger.info("create transformations")
    transformers = create_transformers(
        image_key=data.keys[0],
        mask_key=data.keys[1],
        image_size=data.image_size
    )

    logger.info("creating data loaders")
    loaders = create_data_loaders(data, transformers, batch_sizes=(128, 64), num_workers=cpu_count())

    logger.info("trainer initialization")
    trainer = pl.Trainer(**make_trainer_init_params(params))

    logger.info("training started...")
    trainer.fit(model=ToyExperiment(params),
                train_dataloader=loaders["train"],
                val_dataloaders=loaders["valid"])

    logger.info("saving experiment parameters into checkpoints folder")
    params_dir = save_experiment_info(trainer, {
        "params": params,
        "data": data,
        "transformers": transformers
    })

    if params_dir is None:
        logger.warning("checkpoints dir is not found; will not save info")
    else:
        logger.info("experiment artifacts saved into folder: %s", params_dir)


class ToyExperiment(BaseExperiment):

    def create_model(self) -> nn.Module:
        return create_unet_model(self.hparams)

    def forward(self, batch: Dict) -> Dict:
        predicted_mask = self.model(batch["img"])
        loss = self.loss_fn(predicted_mask, batch["seg"])
        return {"loss": loss, "outputs": predicted_mask}


if __name__ == '__main__':
    main()
