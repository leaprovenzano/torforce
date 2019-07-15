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

    def log_prob(self, sample):
        p = super().log_prob(sample)
        return p.sum(-1)

    def entropy(self):
        ent = super().entropy()
        return ent.sum(-1)

class ScaledUnimodalBeta(UnimodalBeta):

    """Unimodal beta with a scaler attatched that rescales at the following points:

       - outputs to the sample method are rescaled from [0, 1] to the specified output range
       - inputs to log_prob are rescaled from the specified output range back to [0,1]

    Examples:

        Usually you will want to create a ScaledUniModalBeta factory from with your output range.

        >>> import torch
        >>> from torforce.distributions import ScaledUnimodalBeta
        >>> 
        >>> ScaledDist = ScaledUnimodalBeta.from_range((-1, 1))

        ...that was one time setup, now you can use scaled dist as normal with alpha & beta params.
        Now we can instantiate single instances of our ScaledDist with inputs `alpha` and `beta` as normal:

        >>> a = torch.tensor([[2.93], [0.65]])
        >>> b = torch.tensor([[1.40], [0.06]])
        >>
        >>> dist = ScaledDist(a, b)

        our outputs will be properly rescaled from the [0, 1] range of the beta distribution
        to our the range we specified on creation of our `ScaledDist` factory:

        >>> dist.sample()
        tensor([[ 0.2011],
                [-0.6362]])

        the log_prob method will also properly inverse_scale samples before calculating probabilities. 

        >>> dist.log_prob(torch.tensor([[ 0.2011], [-0.6362]]))
        tensor([[ 0.6600],
                [-0.5400]])
    """

    @classmethod
    def from_range(cls, new_range):
        return partial(cls, new_range)

    def __init__(self, output_range, alpha, beta, validate_args=None):
        super().__init__(alpha, beta, validate_args=validate_args)
        self._sample_scaler = MinMaxScaler((0, 1), output_range)

    def sample(self, *args, **kwargs):
        samp = super().sample(*args, **kwargs)
        return self._sample_scaler.scale(samp)

    def log_prob(self, sample, *args, **kwargs):
        return super().log_prob(self._sample_scaler.inverse_scale(sample), *args, **kwargs)


class LogCategorical(Categorical):

    """Categorical distribution from logits by default.
    """

    def __init__(self, logits: torch.Tensor):
        super().__init__(logits=logits)
