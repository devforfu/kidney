from monai.networks.nets import Unet
from pytorch_lightning.utilities import AttributeDict

from kidney.parameters import requires


@requires([
    "unet_dimensions",
    "unet_in_channels",
    "unet_out_channels",
    "unet_layer_sizes",
    "unet_strides",
    "unet_number_residual_units",
    "unet_kernel_size",
    "unet_up_kernel_size"
])
def create_unet_model(params: AttributeDict) -> Unet:
    return Unet(
        dimensions=params.unet_dimensions,
        in_channels=params.unet_in_channels,
        out_channels=params.unet_out_channels,
        channels=params.unet_layer_sizes,
        strides=params.unet_strides,
        num_res_units=params.unet_number_residual_units,
        kernel_size=params.unet_kernel_size,
        up_kernel_size=params.unet_up_kernel_size
    )
