import torch
from torch import nn


class SoftPlusOne(nn.Softplus):

    """this is simply softplus(x) + 1

    Example:
        >>> import torch
        >>> from torforce.activations import SoftPlusOne
        >>>
        >>> act = SoftPlusOne()
        >>> act(torch.tensor(.5))
        tensor(1.9741)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) + 1


_registry = {
    'elu': nn.ELU,
    'hardshrink': nn.Hardshrink,
    'hardtanh': nn.Hardtanh,
    'leakyrelu': nn.LeakyReLU,
    'logsigmoid': nn.LogSigmoid,
    'prelu': nn.PReLU,
    'relu': nn.ReLU,
    'relu6': nn.ReLU6,
    'rrelu': nn.RReLU,
    'selu': nn.SELU,
    'celu': nn.CELU,
    'gelu': nn.GELU,
    'sigmoid': nn.Sigmoid,
    'softplus': nn.Softplus,
    'softshrink': nn.Softshrink,
    'softsign': nn.Softsign,
    'tanh': nn.Tanh,
    'tanhshrink': nn.Tanhshrink,
    'softplus1': SoftPlusOne,
    'softplusone': SoftPlusOne,
}


def get_named_activation(name: str) -> nn.Module:
    """get an instance of an activation module by it's name.

    Note:
        if the activation allows init params it will only ever be initilized
        with the default values.

    Example:
        >>> from torforce.activations import get_named_activation
        >>>
        >>> get_named_activation('relu')
        ReLU()
    """
    try:
        return _registry[name.lower()]()
    except KeyError:
        raise ValueError(f'unknown activation {name}: activation could not be found in registry')
