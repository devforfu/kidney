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
        pretrained_backbone=getattr(params, "fcn_pretrained_backbone", False),
        progress=False,
        aux_loss=getattr(params, "fcn_aux_loss", False),
    )
    n_classes = params.fcn_num_classes
    # TODO: check if resnet101 has the same number of classifier/aux blocks
    model.classifier[4] = nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, n_classes, kernel_size=(1, 1), stride=(1, 1))
    return model
