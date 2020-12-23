import warnings
import wandb


def noop(*args, **kwargs):
    pass


setattr(wandb, 'termwarn', noop)
setattr(warnings, 'warn', noop)
