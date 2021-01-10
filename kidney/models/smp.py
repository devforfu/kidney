import segmentation_models_pytorch as smp
import torch.nn as nn
from pytorch_lightning.utilities import AttributeDict
from segmentation_models_pytorch.encoders import get_preprocessing_fn

from kidney.parameters import get_relevant_params, requires


@requires([
    "smp_model_type",
    "smp_model_config"
])
def create_smp_model(params: AttributeDict) -> nn.Module:
    config = params.smp_model_config
    preprocess_input = get_preprocessing_fn(
        encoder_name=config.get("encoder_name", "resnet34"),
        pretrained=config.get("encoder_weights", "imagenet")
    )
    if params.smp_model_type == "unet":
        params = get_relevant_params(smp.Unet.__init__, config)
        factory = smp.Unet(**params)
    else:
        raise ValueError(f"unknown model type: {params.smp_model_type}")
    return factory, preprocess_input
