from argparse import ArgumentParser

from kidney.cli import extend_parser, comma_separated_list_of_integers
from kidney.datasets.transformers import IntensityNormalization


@extend_parser
def add_model_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--model_input_size", type=int, default=1024)
    parser.add_argument("--model_input_image_key", default="img")
    parser.add_argument("--model_input_mask_key", default="seg")
    return parser


@extend_parser
def add_sliding_window_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--sliding_window_overlap", type=int, default=32)
    parser.add_argument("--sliding_window_outliers_threshold", type=int, default=None)
    return parser


@extend_parser
def add_monai_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--monai_image_key", default="img")
    parser.add_argument("--monai_mask_key", default="seg")
    parser.add_argument("--monai_crop_fraction", type=float, default=0.5)
    parser.add_argument("--monai_crop_num_samples", type=int, default=4)
    parser.add_argument("--monai_rotation_prob", type=float, default=0.5)
    parser.add_argument("--monai_crop_balanced", action="store_true")
    parser.add_argument("--monai_channels_first", action="store_true")
    parser.add_argument("--monai_load_from_disk", action="store_true")
    parser.add_argument("--monai_pos_fraction", type=float, default=1.0)
    parser.add_argument("--monai_neg_fraction", type=float, default=1.0)
    parser.add_argument("--monai_normalization",
                        type=IntensityNormalization,
                        default=IntensityNormalization.TorchvisionSegmentation)
    return parser


@extend_parser
def add_aug_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--aug_pipeline", default="weak")
    parser.add_argument("--aug_normalization_method",
                        type=IntensityNormalization,
                        default=IntensityNormalization.TorchvisionSegmentation)
    return parser


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
    parser.add_argument("--fcn_pretrained_backbone", action="store_true")
    parser.add_argument("--fcn_aux_loss", action="store_true")
    parser.add_argument("--fcn_num_classes", type=int, default=1)
    return parser
