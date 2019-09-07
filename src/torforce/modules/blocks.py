import torch
from torch import nn


class LinearBlock(nn.Module):

    """create a simple linear layer optionally including activation, layer normalization and dropout.

    Args:
        in_features (int): input features
        out_features (int): ouptut features
        layer_norm (bool, optional): if true apply `nn.LayerNormalization` before activation. Defaults to `False`.
        activation (nn.Module, optional): activation. Defaults to None.
        dropout_rate (float, optional): dropout rate. Defaults to 0. (no dropout).
        bias (bool, optional): if true include bias in linear layer. Defaults to `True`.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 layer_norm: bool=False,
                 activation: nn.Module=None,
                 dropout_rate: float=0.,
                 bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if layer_norm:
            self.normalization = nn.LayerNorm(self.out_features)
        self.activation = activation
        if dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.children():
            x = m(x)
        return x
