"""distributions based off of pytorch.distributions for use with policy models.

"""
import torch
from functools import partial
from torch.distributions import Beta, Categorical, Normal
from torforce.utils import MinMaxScaler


class UnimodalBeta(Beta):

    """Adjusted Beta(α + 1, β + 1) which when used along with softplus activation
    ensures our Beta distribution is always unimodal.


    Examples:


        >>> import torch
        >>> from torforce.distributions import UnimodalBeta
        >>>
        >>> a = torch.tensor([[2.93, 0.65, 0.84],
        ...                   [0.52, 3.39, 5.86]])
        >>> b = torch.tensor([[1.40, 0.06, 0.55],
        ...                   [6.45, 3.83, 1.16]])
        >>> dist = UnimodalBeta(a, b)

        >>> dist.concentration1
        tensor([[3.9300, 1.6500, 1.8400],
                [1.5200, 4.3900, 6.8600]])

        >>> dist.concentration0
        tensor([[2.4000, 1.0600, 1.5500],
                [7.4500, 4.8300, 2.1600]])


        >>> dist.sample()
        tensor([[0.4651, 0.3861, 0.5077],
                [0.2856, 0.5835, 0.7325]])

    """

    def __init__(self, alpha, beta, validate_args=None):
        super().__init__(alpha + 1, beta + 1, validate_args=validate_args)



class LogCategorical(Categorical):

    """Categorical distribution from logits by default.
    """

    def __init__(self, logits: torch.Tensor):
        super().__init__(logits=logits)
