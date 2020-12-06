"""A toy experiment using randomly generated images and MONAI models."""
from pytorch_lightning.utilities import AttributeDict

from kidney.cli import default_args, entry_point
from kidney.cli.basic import basic_parser
from kidney.cli.models import add_unet_args


@entry_point(
    base_parser_factory=lambda: basic_parser(__file__),
    extensions=default_args() + [add_unet_args]
)
def main(params: AttributeDict):
    print(params)


if __name__ == '__main__':
    main()
