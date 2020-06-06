import torch
from torch import nn

from torforce.utils.repr_utils import simple_repr


class Flatten(nn.Module):

    """module wrapper for torch.flatten

    Args:
        start_dim (int, optional): first dim to flatten. Defaults to 1.
        end_dim (int, optional): last dim to flatten. Defaults to -1.

    Example:
        >>> import torch
        >>> from torforce.modules import Flatten
        >>>
        >>> flat = Flatten()
        >>> flat
        Flatten(start_dim=1, end_dim=-1)

        >>> x = torch.rand(5, 3, 3)
        >>> flat(x).shape
        torch.Size([5, 9])
    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, start_dim=self.start_dim, end_dim=self.end_dim)

    def __repr__(self) -> str:
        return simple_repr(self, start_dim=self.start_dim, end_dim=self.end_dim)
