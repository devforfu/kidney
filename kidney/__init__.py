import warnings

import torch
import wandb


def noop(*args, **kwargs):
    pass


setattr(wandb, 'termwarn', noop)
setattr(warnings, 'warn', noop)
torch.set_default_tensor_type(torch.FloatTensor)
