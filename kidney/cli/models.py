from argparse import ArgumentParser

from kidney.cli import extend_parser, comma_separated_list_of_integers


@extend_parser
def add_unet_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--unet_dimensions", type=int, default=2)
    parser.add_argument("--unet_in_channels", type=int, default=3)
    parser.add_argument("--unet_out_channels", type=int, default=1)
    parser.add_argument("--unet_layer_sizes",
                        type=comma_separated_list_of_integers,
                        default=[16, 32, 64, 128, 256, 512])
    parser.add_argument("--unet_strides",
                        type=comma_separated_list_of_integers,
                        default=[2, 2, 2, 2])
    parser.add_argument("--unet_number_residual_units", type=int, default=2)
    parser.add_argument("--unet_kernel_size", type=int, default=3)
    parser.add_argument("--unet_up_kernel_size", type=int, default=3)
    return parser


@extend_parser
def add_fcn_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--fcn_arch", choices=["resnet50", "resnet101"], default="resnet50")
    parser.add_argument("--fcn_pretrained", action="store_true")
    parser.add_argument("--fcn_num_classes", type=int, default=1)
    return parser
