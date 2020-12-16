from typing import Dict

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.utilities import AttributeDict
from zeus.core.random import super_seed

from kidney.cli import entry_point, default_args
from kidney.cli.basic import basic_parser
from kidney.cli.lightning import make_trainer_init_params
from kidney.cli.models import add_unet_args
from kidney.datasets.segmentation import read_segmentation_data_from_json, create_data_loaders
from kidney.datasets.transformers import create_transformers_crop_to_many
from kidney.experiments import BaseExperiment, save_experiment_info
from kidney.log import get_logger
from kidney.models.unet import create_unet_model
from kidney.utils.image import random_image_shape


@entry_point(
    base_parser_factory=lambda: basic_parser(__file__),
    extensions=default_args() + [add_unet_args]
)
def main(params: AttributeDict):
    super_seed(params.seed)

    logger = get_logger(__file__)

    logger.info("reading segmentation images")
    json_file = f"{params.dataset}/histograms.json"
    data = read_segmentation_data_from_json(json_file)

    logger.info("create transformers")
    transformers = create_transformers_crop_to_many(
        image_key=data.image_key,
        mask_key=data.mask_key,
        image_size=random_image_shape(params.dataset),
        crop_balanced=False
    )

    logger.info("creating data loaders")
    loaders = create_data_loaders(
        data=data,
        transformers=transformers,
        batch_size=params.batch_size,
        num_workers=params.num_workers
    )

    logger.info("trainer initialization")
    trainer = pl.Trainer(**make_trainer_init_params(params))

    logger.info("training started...")
    trainer.fit(model=SegmentationExperiment(params),
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


class SegmentationExperiment(BaseExperiment):

    def create_model(self) -> nn.Module:
        return create_unet_model(self.hparams)

    def forward(self, batch: Dict) -> Dict:
        predicted_mask = self.model(batch["img"])
        loss = self.loss_fn(predicted_mask, batch["seg"])
        return {"loss": loss, "outputs": predicted_mask}


if __name__ == '__main__':
    main()
