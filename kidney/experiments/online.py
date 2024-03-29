import pytorch_lightning as pl
from pytorch_lightning.utilities import AttributeDict
from zeus.core.random import super_seed

from kidney.cli import entry_point, default_args
from kidney.cli.basic import basic_parser
from kidney.cli.lightning import make_trainer_init_params
from kidney.cli.models import add_fcn_args, add_model_args, add_monai_args
from kidney.datasets.kaggle import get_reader
from kidney.datasets.offline import create_data_loaders
from kidney.datasets.transformers import create_monai_crop_to_many_sigmoid_transformers
from kidney.datasets.utils import read_boxes
from kidney.experiments import save_experiment_info, FCNExperiment
from kidney.log import get_logger


@entry_point(
    base_parser_factory=lambda: basic_parser(__file__),
    extensions=default_args() + [
        add_fcn_args,
        add_model_args,
        add_monai_args
    ]
)
def main(params: AttributeDict):
    super_seed(params.seed)

    logger = get_logger(__file__)

    logger.info("creating dataset reader")
    reader = get_reader()

    logger.info("creating transformers")
    assert 0 < params.monai_crop_fraction < 1

    transformers = create_monai_crop_to_many_sigmoid_transformers(
        image_key=params.monai_image_key,
        mask_key=params.monai_mask_key,
        image_size=params.model_input_size,
        crop_fraction=params.monai_crop_fraction,
        crop_balanced=params.monai_crop_balanced,
        num_samples=params.monai_crop_num_samples,
        rotation_prob=params.monai_rotation_prob,
        load_from_disk=params.monai_load_from_disk,
        as_channels_first=params.monai_channels_first,
        normalization=params.monai_normalization,
        pos_fraction=params.monai_pos_fraction,
        neg_fraction=params.monai_neg_fraction
    )

    logger.info("creating data loader")
    loaders = create_data_loaders(
        reader=reader,
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


if __name__ == '__main__':
    main()
