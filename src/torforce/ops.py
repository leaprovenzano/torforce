from typing import Union, Tuple
import torch
from torforce.distributions import TorforceDistributionMixin


def expanded_empty_like(
    x: Union[torch.Tensor, TorforceDistributionMixin], new_shape: Tuple[int],
):
    """get a new empty tensorlike object from the given tensor or tensorlike object with \
    new expanded shape.

    Example:
        >>> from torforce.ops import expanded_empty_like
        >>>
        >>> x = torch.rand(3, 4)
        >>> new_empty = expanded_empty_like(x, (5, ...))
        >>> new_empty.shape
        torch.Size([5, 3, 4])

        >>> new_empty = expanded_empty_like(x, (5, ..., 1))
        >>> new_empty.shape
        torch.Size([5, 3, 4, 1])

        >>> from torforce.distributions import IndyBeta
        >>>
        >>> original_dist = IndyBeta(torch.rand(3, 4), torch.rand(3, 4))
        >>> new_empty_dist = expanded_empty_like(original_dist, (10, ...))
        >>> new_empty_dist.shape
        torch.Size([10, 3, 4])

        >>> new_empty_dist.batch_ndims
        2

        >>> type(new_empty_dist)
        <class 'torforce.distributions.IndyBeta'>

    """
    shape: Tuple[int] = tuple([])  # type: ignore
    for i in new_shape:
        # if we encounter an elipsis replace with the whole shape of
        # the input tensor
        shape = shape + x.shape if i is Ellipsis else shape + (i,)  # type: ignore

    if isinstance(x, TorforceDistributionMixin):
        params = [torch.empty(shape, dtype=p.dtype, device=p.device) for p in x.params]
        return x.__class__(*params)

    return torch.empty(tuple(shape), dtype=x.dtype, device=x.device)
