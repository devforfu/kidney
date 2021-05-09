from itertools import filterfalse

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def flatten_binary_scores(scores, labels, ignore=None):
    """Flattens predictions in the batch (binary case).

    Remove labels equal to 'ignore'.
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    v_scores = scores[valid]
    v_labels = labels[valid]
    return v_scores, v_labels


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """Binary Lovasz hinge loss.

    Parameters
    ----------
    logits:
        Logits at each pixel [B, H, W].
    labels:
        Binary ground truth masks (0 or 1) [B, H, W].
    per_image
        Compute the loss per image instead of per batch
    ignore:
        Void class ID

    """
    if per_image:
        loss = mean(
            lovasz_hinge_flat(
                *flatten_binary_scores(
                    log.unsqueeze(0),
                    lab.unsqueeze(0),
                    ignore
                )
            )
            for log, lab in zip(logits, labels)
        )
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss.

    Parameters
    ----------
    logits
        Logits at each prediction.
    labels
        Binary ground truth labels (0 or 1).
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def isnan(x):
    return x != x


def mean(values, ignore_nan=False, empty=0):
    values = iter(values)
    if ignore_nan:
        values = filterfalse(isnan, values)
    try:
        n = 1
        acc = next(values)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(values, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def main():
    loss = lovasz_hinge
    o1 = torch.tensor([[
        [-3.0, -2.0, 0.1, 1.3],
        [-5.1, -3.5, -2.0, 1.0],
        [1.2, 0.7, -2.3, -1.0],
        [0.1, 0.2, 0.1, 0.4],
    ]])
    o2 = torch.tensor([[
        [-30.0, -30.0, 30.0, 30.0],
        [-30.0, -30.0, -30.0, 30.0],
        [30.0, -30.0, -30.0, -30.0],
        [30.0, 30.0, 30.0, 30.0],
    ]])
    o3 = torch.tensor([[
        [-15.0, -5.0, 3.0, 10.0],
        [-10.0, -3.0, -1.0, 3.0],
        [1.0, -1.0, -5.0, -3.0],
        [5.0, 1.0, 0.1, 3.0],
    ]])
    t1 = torch.tensor([[
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]])
    t2 = torch.tensor([[
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]])
    t3 = torch.tensor([[
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]])
    t4 = torch.tensor([[
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ]])
    print(loss(o1, t1))
    print(loss(o1, t2))
    print(loss(o1, t3))
    print(loss(o1, t4))
    print(loss(o2, t1))
    print(loss(o2, t2))
    print(loss(o2, t3))
    print(loss(o2, t4))
    print(loss(o3, t1))
    print(loss(o3, t2))
    print(loss(o3, t3))
    print(loss(o3, t4))


if __name__ == '__main__':
    main()
