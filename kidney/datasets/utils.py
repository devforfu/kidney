import os


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
