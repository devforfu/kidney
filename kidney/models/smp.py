import segmentation_models_pytorch as smp
import torch.nn as nn
from pytorch_lightning.utilities import AttributeDict

from kidney.parameters import get_relevant_params, requires


@requires([
    "smp_model_type",
    "smp_model_config"
])
def create_smp_model(params: AttributeDict) -> nn.Module:
    if params.smp_model_type == "unet":
        unet_params = get_relevant_params(smp.Unet.__init__, params.smp_model_config)
        return smp.Unet(**unet_params)
    else:
        raise ValueError(f"unknown model type: {params.smp_model_type}")
