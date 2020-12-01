import glob
from collections import defaultdict
from operator import itemgetter

from zeus.utils import named_match
from zeus.utils.collections import NamedList

DEFAULT_REGEX = (
    r'(?P<key>[\w\d]+)\.'
    r'(?P<dx>\d+)\.'
    r'(?P<dy>\d+)\.'
    r'(?P<stride>\d+)'
)


def read_masked_images(
    folder: str,
    image_suffix: str = 'img',
    mask_suffix: str = 'seg',
    file_regex: str = DEFAULT_REGEX,
    image_extension: str = 'png'
) -> NamedList:
    images_info = defaultdict(dict)

    for suffix in (image_suffix, mask_suffix):
        pattern = rf'{suffix}\.{file_regex}\.{image_extension}'

        for path in glob.glob(f'{folder}/{suffix}.*.{image_extension}'):
            m = named_match(pattern, path)
            key, dx, dy = m['key'], m['dx'], m['dy']
            identifier = f'{key}.{dx}.{dy}'
            images_info[identifier][
                'mask'
                if suffix == mask_suffix
                else 'image'
            ] = path
            images_info[identifier]['position'] = dx, dy

    sorted_keys = map(
        itemgetter(0),
        sorted(images_info.items(), key=itemgetter(0))
    )

    return NamedList([(key, images_info[key]) for key in sorted_keys])
