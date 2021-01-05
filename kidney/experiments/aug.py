import os

import pytorch_lightning as pl
from pytorch_lightning.utilities import AttributeDict
from zeus.core.random import super_seed

from kidney.cli import entry_point, default_args
from kidney.cli.basic import basic_parser
from kidney.cli.lightning import make_trainer_init_params
from kidney.cli.models import add_fcn_args, add_model_args, add_aug_args
from kidney.datasets.kaggle import get_reader
from kidney.datasets.online import create_data_loaders, read_boxes
from kidney.datasets.transformers import create_weak_augmentation_transformers
from kidney.experiments import save_experiment_info, FCNExperiment
from kidney.log import get_logger


@entry_point(
    base_parser_factory=lambda: basic_parser(__file__),
    extensions=default_args() + [
        add_fcn_args,
        add_model_args,
        add_aug_args,
    ]
)
def main(params: AttributeDict):
    super_seed(params.seed)

    logger = get_logger(__file__)
    input_image_size = get_dataset_input_size(params.dataset)

    logger.info("creating dataset reader")
    reader = get_reader()

    logger.info("creating transformers")
    transformers = create_weak_augmentation_transformers(
        image_key=params.model_input_image_key,
        mask_key=params.model_input_mask_key,
        image_size=input_image_size,
        normalization=params.aug_normalization_method
    )

    logger.info("creating data loaders")
    loaders = create_data_loaders(
        reader=reader,
        valid_keys=[params.fold] if "fold" in params else None,
        transformers=transformers,
        samples=read_boxes(params.dataset),
        num_workers=params.num_workers,
        batch_size=params.batch_size,
    )

    trainer = pl.Trainer(**make_trainer_init_params(params))

    save_experiment_info(trainer, {"params": params, "transformers": transformers})

    trainer.fit(model=FCNExperiment(params),
                train_dataloader=loaders["train"],
                val_dataloaders=loaders["valid"])


def get_dataset_input_size(path: str) -> int:
    _, folder = os.path.split(path)
    try:
        crop_size = int(folder.split("_")[-1])
        return crop_size
    except TypeError:
        raise RuntimeError(f"cannot parse input image size from path string: {path}")


if __name__ == '__main__':
    main()
