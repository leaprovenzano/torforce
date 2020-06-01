"""grad and modeling utils"""

from contextlib import contextmanager
import functools

import torch
from torch import nn


def freeze(model: nn.Module):
    """Freeze all parameters in the given model.
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model: nn.Module):
    """Un-freeze all parameters in the given model.
    """
    for param in model.parameters():
        param.requires_grad = True


@contextmanager
def eval_context(model: nn.Module):
    """context manager for temporarily setting model in eval mode.

    Example:
        >>> from torforce.modules.utils import eval_context
        >>>
        >>> model = nn.Sequential(nn.Linear(5, 10), nn.Linear(10, 10))
        >>> with eval_context(model):
        ...     print(model.training)
        False

        >>> model.training
        True
    """
    was_train = model.training
    try:
        model.eval()
        yield model
    finally:
        if was_train:
            model.train()


def eval_mode(f):
    """a decorator designed for nn.Module methods which wraps the function call in the eval context.

    Note:
        you can use this decorator for any function that takes a model as first parameter.
        but it's reccommended to use on nn.Module methods.
    """

    @functools.wraps(f)
    def inner(model, *args, **kwargs):
        with eval_context(model):
            return f(model, *args, **kwargs)

    return inner


def no_grad_mode(f):
    """a decorator designed for nn.Module methods -- wraps the function call in the no-grad context.

    Note:
        you can use this decorator for any function that takes a model as first parameter.
        but it's reccommended to use on nn.Module methods.
    """

    @functools.wraps(f)
    def inner(model, *args, **kwargs):
        with torch.no_grad():
            return f(model, *args, **kwargs)

    return inner
