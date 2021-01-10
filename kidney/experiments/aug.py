import pytorch_lightning as pl
from pytorch_lightning.utilities import AttributeDict
from zeus.core.random import super_seed

from kidney.cli import entry_point, default_args
from kidney.cli.basic import basic_parser
from kidney.cli.lightning import make_trainer_init_params
from kidney.cli.models import add_fcn_args, add_model_args, add_aug_args
from kidney.cli.training import add_validation_args
from kidney.datasets.kaggle import get_reader
from kidney.datasets.online import create_data_loaders, read_boxes
from kidney.datasets.transformers import get_transformers
from kidney.experiments import save_experiment_info, FCNExperiment
from kidney.log import get_logger


@entry_point(
    base_parser_factory=lambda: basic_parser(__file__),
    extensions=default_args() + [
        add_fcn_args,
        add_model_args,
        add_aug_args,
        add_validation_args
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
        valid_keys = [params.fold] if fold_training else None
        params["fold"] = params.fold
    else:
        valid_keys = None

    logger.info("creating transformers")
    transformers = get_transformers(params)

    logger.info("creating data loaders")
    loaders = create_data_loaders(
        reader=reader,
        valid_keys=valid_keys,
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
