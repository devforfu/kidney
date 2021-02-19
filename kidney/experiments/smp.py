from typing import Dict, List

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.utilities import AttributeDict
from zeus.core.random import super_seed

from kidney.cli import entry_point, default_args
from kidney.cli.basic import basic_parser
from kidney.cli.dynamic import add_dynamically_defined_params
from kidney.cli.lightning import make_trainer_init_params
from kidney.cli.models import add_model_args, add_aug_args, add_smp_args
from kidney.cli.training import add_validation_args, add_data_loader_extra_args
from kidney.datasets.kaggle import get_reader
from kidney.datasets.offline import create_data_loaders
from kidney.datasets.transformers import get_transformers
from kidney.datasets.utils import read_segmentation_info
from kidney.experiments import save_experiment_info, BaseExperiment
from kidney.log import get_logger
from kidney.models.smp import create_smp_model


@entry_point(
    base_parser_factory=lambda: basic_parser(__file__),
    extensions=default_args() + [
        add_smp_args,
        add_model_args,
        add_aug_args,
        add_validation_args,
        add_data_loader_extra_args,
        add_dynamically_defined_params([
            {"name": "--file_format",
             "default": "bbox",
             "choices": ["bbox", "enum"],
             "help": "Dataset files format"}
        ])
    ]
)
def main(params: AttributeDict):
    super_seed(params.seed)

    logger = get_logger(__file__)
    fold_training = params.get("fold") is not None

    logger.info("creating dataset reader")
    reader = get_reader()

    if fold_training:
        logger.info(f"running in K-fold training mode; the current fold: {params.fold}")
        params["fold"] = params.fold
        valid_keys = parse_fold_keys(params.fold)
    else:
        valid_keys = None

    logger.info("creating transformers")
    transformers = get_transformers(params)

    logger.info("creating data loaders")
    loaders = create_data_loaders(
        reader=reader,
        valid_keys=valid_keys,
        transformers=transformers,
        samples=read_segmentation_info(params.dataset, file_format=params.file_format),
        num_workers=params.num_workers,
        batch_size=params.batch_size,
        multiprocessing_context=params.data_loader_multiprocessing_context
    )

    trainer = pl.Trainer(**make_trainer_init_params(params))

    save_experiment_info(trainer, {"params": params, "transformers": transformers})

    trainer.fit(model=SMPExperiment(params),
                train_dataloader=loaders["train"],
                val_dataloaders=loaders["valid"])


def parse_fold_keys(fold: str) -> List[str]:
    return fold.split(";") if ";" in fold else [fold]


class SMPExperiment(BaseExperiment):

    def create_model(self) -> nn.Module:
        return create_smp_model(self.hparams)

    def forward(self, batch: Dict) -> Dict:
        predicted_mask = self.model(batch["img"])
        if "seg" in batch:
            loss = self.loss_fn(predicted_mask, batch["seg"])
            if isinstance(loss, Dict):
                loss["outputs"] = predicted_mask
                return loss
            else:
                return {"loss": loss, "outputs": predicted_mask}
        return {"outputs": predicted_mask}


if __name__ == '__main__':
    main()
