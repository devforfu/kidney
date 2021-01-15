from typing import Any, Dict

import segmentation_models_pytorch as smp
import torch.nn as nn
from pytorch_lightning.utilities import AttributeDict

from kidney.parameters import get_relevant_params, requires

_MODEL_FACTORY = {
    "unet": smp.Unet,
    "unet++": smp.UnetPlusPlus,
}


@requires([
    "smp_model_type",
    "smp_model_config"
])
def create_smp_model(params: AttributeDict) -> nn.Module:
    model_name = params.smp_model_type.lower()
    model_cls = key_or_raise(_MODEL_FACTORY, model_name)
    params = get_relevant_params(model_cls.__init__, params.smp_model_config)
    model = model_cls(**params)
    return model


def key_or_raise(d: Dict, key: str) -> Any:
    if key not in d:
        keys = sorted(list(d.keys()))
        raise KeyError(f"key is not found: {key}; available keys are: {keys}")
    return d[key]
