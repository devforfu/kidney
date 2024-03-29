import albumentations as A
import copy
from albumentations.pytorch import ToTensorV2

from kidney.datasets.transformers import (
    create_normalization,
    add_first_channel,
    SigmoidOutputAsMask,
    IntensityNormalization,
    Transformers,
)
from prototype.config import TransformersConfig


def create_transformers(
    config: TransformersConfig,
    normalization: IntensityNormalization = IntensityNormalization.TorchvisionSegmentation,
    debug: bool = False,
) -> Transformers:

    train_steps, valid_steps = [], []

    if config.resize:
        image_size = config.image_size
        if image_size is not None:
            train_steps.append(A.Resize(image_size, image_size))
            valid_steps.append(A.Resize(image_size, image_size))

    if not config.normalize_only:

        if config.bcg is not None:
            bcg = config.bcg
            train_steps.append(
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(bcg.brightness, bcg.contrast, p=1),
                        A.RandomGamma(bcg.gamma, p=1),
                    ],
                    p=bcg.prob,
                )
            )

        if config.blur is not None:
            blur = config.blur
            train_steps.append(
                A.OneOf(
                    [A.Blur(blur_limit=blur.limit), A.MedianBlur(blur_limit=blur.limit)],
                    p=blur.prob,
                )
            )

        if config.noise is not None:
            noise = config.noise
            train_steps.append(
                A.OneOf(
                    [A.GaussNoise(var_limit=noise.gauss_var)]
                    if noise.affine_scale is None
                    else [
                        A.GaussNoise(var_limit=noise.gauss_var),
                        A.IAAAffine(scale=noise.affine_scale),
                    ],
                    p=noise.prob,
                )
            )

        if config.flip is not None:
            flip = config.flip
            flips = []
            if flip.vertical:
                flips.append(A.VerticalFlip(p=flip.prob))
            if flip.horizontal:
                flips.append(A.HorizontalFlip(p=flip.prob))
            train_steps.extend(flips)

        if config.elastic is not None:
            elastic = config.elastic
            train_steps.append(
                A.ElasticTransform(
                    alpha=elastic.alpha,
                    sigma=elastic.sigma,
                    alpha_affine=elastic.alpha_affine,
                    p=elastic.prob,
                )
            )

        if config.ssr is not None:
            ssr = config.ssr
            train_steps.append(
                A.ShiftScaleRotate(
                    shift_limit=ssr.shift,
                    scale_limit=ssr.scale,
                    rotate_limit=ssr.rotate,
                    p=ssr.prob,
                )
            )

        if config.color is not None:
            color_steps = []
            if config.color.hsv is not None:
                color_steps.append(
                    A.HueSaturationValue(
                        hue_shift_limit=config.color.hsv.hue,
                        sat_shift_limit=config.color.hsv.saturation,
                        val_shift_limit=config.color.hsv.value,
                        p=config.color.hsv.prob
                    )
                )
            if config.color.hist is not None:
                color_steps.append(
                    A.CLAHE(
                        clip_limit=config.color.hist.clip,
                        tile_grid_size=config.color.hist.tile_grid,
                        p=config.color.hist.prob
                    )
                )
            if config.color.brightness_contrast is not None:
                color_steps.append(
                    A.RandomBrightnessContrast(
                        brightness_limit=config.color.brightness_contrast.brightness,
                        contrast_limit=config.color.brightness_contrast.contrast,
                        p=config.color.brightness_contrast.prob
                    )
                )
            train_steps.append(A.OneOf(color_steps, p=config.color.prob))

    final_step = (
        [A.NoOp()]
        if debug
        else [
            create_normalization(normalization, skip=debug),
            A.Lambda(name="add_channel_axis", mask=add_first_channel),
            ToTensorV2(),
        ]
    )

    train_steps.extend(final_step)
    valid_steps.extend(copy.deepcopy(final_step))

    return Transformers(
        train=A.Compose(train_steps),
        valid=A.Compose(valid_steps),
        test_preprocessing=A.Compose(valid_steps),
        test_postprocessing=SigmoidOutputAsMask(),
    )
