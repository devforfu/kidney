import cv2 as cv
import numpy as np


def remove_overlap(mask: np.ndarray, n_dims: int = 2) -> np.ndarray:
    classes = np.unique(mask)[1:]

    instance_labels = np.zeros_like(mask)
    next_instance = 1
    for c in classes:
        class_mask = (mask == c).astype(np.uint8)
        n_instances, components = cv.connectedComponents(class_mask, connectivity=4)
        non_zero = components > 0
        instance_labels[non_zero] = components[non_zero] + next_instance
        next_instance += (n_instances - 1)

    labels = np.zeros_like(mask)

    kernel = np.ones((3,) * n_dims)

    for c in classes:
        class_mask = mask == c
        labels_per_class = (instance_labels * class_mask).astype(np.int16)

        closed = cv.morphologyEx(labels_per_class, cv.MORPH_CLOSE, kernel=kernel)
        overlap = np.unique(np.where(closed != labels_per_class, closed, 0))
        labels[np.isin(labels_per_class, overlap, invert=True)] = c

        for instance in overlap[1:]:
            dilated = cv.dilate((labels == c).astype(np.uint8), kernel=kernel, iterations=1)
            labels[(instance_labels == instance) & (dilated == 0)] = c

    return labels


def create_pdf(mask: np.ndarray, fbr: float, scale: int) -> np.ndarray:
    pdf = (mask > 0) + (mask == 0)*fbr

    if scale:
        if pdf.shape[0] > scale:
            scale_w = int((pdf.shape[1]/pdf.shape[0])*scale)
            pdf = cv.resize(pdf, dsize=(scale_w, scale), interpolation=cv.INTER_CUBIC)

    return np.cumsum(pdf/np.sum(pdf))


def random_center(pdf, orig_shape, scale=512):
    scale_y = int((orig_shape[1]/orig_shape[0])*scale)
    cx, cy = np.unravel_index(np.argmax(pdf > np.random.random()), (scale, scale_y))
    cx = int(cx*orig_shape[0]/scale)
    cy = int(cy*orig_shape[1]/scale_y)
    return cx, cy
