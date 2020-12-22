import matplotlib.pyplot as plt
import numpy as np

from kidney.utils.image import channels_last


def show(image: np.ndarray):
    plt.imshow(channels_last(image))
    plt.show()

