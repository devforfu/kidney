from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class BinaryFocalLoss(_Loss):
    def __init__(
        self, alpha=None, gamma=2, ignore_index=None, reduction="mean", normalized=False, reduced_threshold=None
    ):
        """
        :param alpha: Prior probability of having positive value in target.
        :param gamma: Power factor for dampening weight (focal strenght).
        :param ignore_index: If not None, targets may contain values to be ignored.
        Target values equal to ignore_index will be ignored from loss computation.
        :param reduced: Switch to reduced focal loss. Note, when using this mode you should use `reduction="sum"`.
        :param threshold:
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )

    def forward(self, label_input, label_target):
        """Compute focal loss for binary classification problem."""
        label_target = label_target.view(-1)
        label_input = label_input.view(-1)

        if self.ignore_index is not None:
            # Filter predictions with ignore label from loss computation
            not_ignored = label_target != self.ignore_index
            label_input = label_input[not_ignored]
            label_target = label_target[not_ignored]

            # This check is to prevent NaNs in the loss, when all targets are equal to ignore_index.
            if label_target.numel() == 0:
                return torch.tensor(0, dtype=label_input.dtype, device=label_input.device)

        loss = self.focal_loss_fn(label_input, label_target)
        return loss


def focal_loss_with_logits(
    output: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = 0.25,
    reduction: str = "mean",
    normalized: bool = False,
    reduced_threshold: Optional[float] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute binary focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.
    Args:
        output: Tensor of arbitrary shape (predictions of the model)
        target: Tensor of the same shape as input
        gamma: Focal loss power factor
        alpha: Weight factor to balance positive and negative samples. Alpha must be in [0...1] range,
            high values will give more weight to positive class.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).
    References:
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    target = target.type_as(output)

    logpt = F.binary_cross_entropy_with_logits(output, target, reduction="none")
    pt = torch.exp(-logpt)

    # compute the loss
    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1

    loss = focal_term * logpt

    if alpha is not None:
        loss *= alpha * target + (1 - alpha) * (1 - target)

    if normalized:
        norm_factor = focal_term.sum(dtype=torch.float32).clamp_min(eps)
        loss /= norm_factor

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum(dtype=torch.float32)
    if reduction == "batchwise_mean":
        loss = loss.sum(dim=0, dtype=torch.float32)

    return loss
