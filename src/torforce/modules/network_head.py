from __future__ import annotations
from torch import nn

nn.Identity

from .utils import get_output_shape


class NetworkHead(nn.Module):
    """Base class for policy and value network heads.
    """

    def __init__(self, in_features: int, out_features: int, hidden: nn.Module =nn.Identity()):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden = hidden

        self._hidden_out = get_output_shape(self.hidden, (self.in_features,))[-1]
        self.output_layer = nn.Linear(self._hidden_out, out_features)

    @property
    def info(self):
        return dict(cls=self.__class__.__name__,
                    input_dim=self.input_dim,
                    output_dim=self.output_dim,
                    )

    def activation(self, x):
        return x

    def forward(self, inp):
        x = self.hidden(inp)
        x = self.output_layer(x)
        x = self.activation(x)
        return x
