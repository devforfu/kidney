import os
import random
from collections import OrderedDict
from dataclasses import dataclass
from functools import reduce
from itertools import chain, product
from typing import List, Dict, Set, Optional, Tuple, Callable

from monai.data import list_data_collate
from torch.utils.data import DataLoader, Dataset
from zeus.utils import list_files

from kidney.datasets.transformers import Transformers


def get_dataset_input_size(path: str) -> int:
    """Derives dataset samples size from its path name.

    Convenience helper that works with specifically named folders.

    Parameters
    ----------
    path

    Returns
    -------
    int
        The size of sample. (Rectangular shape is presumed).
    """
    _, folder = os.path.split(path)
    try:
        crop_size = int(folder.split("_")[-1])
        return crop_size
    except TypeError:
        raise RuntimeError(f"cannot parse input image size from path string: {path}")


def read_boxes(folder: str) -> List[Dict]:
    """Reads pre-generated sliding window boxes from JSONL files."""

    import srsly
    return list(chain(*[srsly.read_jsonl(fn) for fn in list_files(folder)]))


def read_segmentation_info(folder: str, file_format: str = "bbox") -> List[Dict]:
    """Creates a list of segmentation dataset samples.

    The folder should include a plain list of PNG files with images and their
    segmentation masks cropped from the original large-scale images. Each file
    should have a unique identifier helping to match image and its mask,
    as well as to extract additional information about sample.

    Example:

        /folder
            /img.<key>_<bbox>.png
            /seg.<key>_<bbox>.png
            ...

    Another accepted format:

        /folder
            /img.<key>_<i>.png
            /seg.<key>_<i>.png

    """
    def get_identifier(filename: str) -> str:
        name, _ = os.path.splitext(filename)
        identifier = name.replace("img.", "").replace("seg.", "")
        return identifier

    unique_samples = sorted({get_identifier(fn) for fn in os.listdir(folder)})

    discovered = []

    for i, sample in enumerate(unique_samples):
        info = {"index": i}
        if file_format == "bbox":
            key, *bbox = sample.split("_")
            bbox = [int(x) for x in bbox]
            info.update(dict(box=bbox, key=key))
        elif file_format == "enum":
            key, num = sample.split("_")
            info.update(dict(num=int(num), key=key))
        info["img"] = os.path.join(folder, f"img.{sample}.png")
        seg_file = os.path.join(folder, f"seg.{sample}.png")
        if os.path.exists(seg_file):
            info["seg"] = seg_file
        discovered.append(info)

    return discovered


def check_disjoint_subsets(superset: Set, *subsets: Set):
    """Checks if given `subsets` are true subsets of `origin` and don't intersect
    with each other.

    Note that this function becomes very computationally expensive as the number of
    subsets increases.
    """
    joined = reduce(lambda x, y: set.union(x, y), subsets, set())
    if superset != joined:
        raise ValueError("the union of subsets is not equal to the superset")
    for a, b in product(subsets, subsets):
        if a is b:
            continue
        if a.intersection(b):
            raise ValueError(f"subsets are not disjoint: {a} and {b}")


def train_valid_keys_split(
    all_keys: List[str],
    train_keys: Optional[List[str]] = None,
    valid_keys: Optional[List[str]] = None,
    sample_size: int = 1
) -> Tuple[List[str], List[str]]:
    """Splits list of string keys into training and valid subsets.

    In case if one of the subsets provided, the other one is created of
    keys that aren't present in the former. Otherwise, `sample_size` keys
    are randomly selected from `keys` and assigned to validation while the
    others are put into training subset.

    Parameters
    ----------
    all_keys
        List of all available keys.
    train_keys
        Training keys.
    valid_keys
        Validation keys.
    sample_size
        If both training and validation keys are None, this number of
        keys is sampled from `all_keys` dictionary and assigned to
        validation while every other key is assigned to training.

    Returns
    -------
    Tuple
        Training and validation keys subsets.

    """
    keys = all_keys
    if train_keys is None and valid_keys is None:
        valid_keys = [random.choice(keys) for _ in range(sample_size)]
        train_keys = [key for key in keys if key not in valid_keys]
    elif train_keys is None and valid_keys is not None:
        train_keys = [key for key in keys if key not in valid_keys]
    elif train_keys is not None and valid_keys is None:
        valid_keys = [key for key in keys if key not in train_keys]
    return train_keys, valid_keys


@dataclass
class KeyedSubset:
    name: str
    keys: Set[str]
    transformer: Callable
    shuffle: bool = False


def create_subset_data_loaders(
    samples: List[Dict],
    subsets: List[KeyedSubset],
    dataset_factory: Callable[[str, List[Dict]], Dataset],
    num_workers: int = 0,
    batch_size: int = 4,
    collate_fn: Callable = list_data_collate,
    **extra
) -> OrderedDict:
    loaders = OrderedDict()
    for subset in subsets:
        samples_subset = [sample for sample in samples if sample["key"] in subset.keys]
        loaders[subset.name] = DataLoader(
            dataset=dataset_factory(subset.name, samples_subset),
            batch_size=batch_size,
            shuffle=subset.shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            **extra
        )
    return loaders


def create_train_valid_data_loaders(
    keys: List[str],
    transformers: Transformers,
    samples: List[Dict],
    dataset_factory: Callable[[List[Dict]], Dataset],
    train_keys: Optional[List[str]] = None,
    valid_keys: Optional[List[str]] = None,
    num_workers: int = 0,
    batch_size: int = 4,
    **extra
):
    train_keys, valid_keys = train_valid_keys_split(keys, train_keys, valid_keys)
    check_disjoint_subsets(set(keys), set(train_keys), set(valid_keys))
    return create_subset_data_loaders(
        samples=samples,
        subsets=[
            KeyedSubset(
                name="train",
                keys=train_keys,
                transformer=transformers.train,
                shuffle=True
            ),
            KeyedSubset(
                name="valid",
                keys=valid_keys,
                transformer=transformers.valid,
                shuffle=False
            )
        ],
        dataset_factory=dataset_factory,
        num_workers=num_workers,
        batch_size=batch_size,
        **extra
    )
