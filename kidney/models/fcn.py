import torch.nn as nn
from pytorch_lightning.utilities import AttributeDict
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101

from kidney.parameters import requires


@requires([
    "fcn_arch",
    "fcn_pretrained",
    "fcn_num_classes"
])
def create_fcn_model(params: AttributeDict) -> nn.Module:
    factory = {
        "resnet50": fcn_resnet50,
        "resnet101": fcn_resnet101
    }[params.fcn_arch]
    model = factory(
        pretrained=params.fcn_pretrained,
        progress=False,
        num_classes=params.fcn_num_classes
    )
    return model
