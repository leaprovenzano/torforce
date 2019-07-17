import numpy as np
import torch

from hypothesis import strategies as st
from hypothesis import given
from hypothesis.strategies import composite, floats

from hypothesis.extra.numpy import arrays, array_shapes
from tests.strategies.torchtensors import float_tensors, variable_batch_shape

from torforce.distributions import UnimodalBeta, LogCategorical


EPS = 1e-5


@composite
def alpha_beta(draw):
    alpha = draw(float_tensors(dtypes='float32', shape=array_shapes(min_dims=1, max_dims=2, min_side=2, max_side=10),
                               unique=True, elements=floats(min_value=0.0000001, max_value=100)))
    beta = draw(float_tensors(dtypes='float32', shape=alpha.shape,
                              unique=True, elements=floats(min_value=0, max_value=100)))
    return alpha, beta


class TestUnimodalBeta(object):

    @given(alpha_beta())
    def test_alpha_beta_gte_one(self, params):
        alpha, beta = params
        dist = UnimodalBeta(alpha, beta)
        np.testing.assert_array_equal(dist.concentration0 >= 1, 1)
        np.testing.assert_array_equal(dist.concentration1 >= 1, 1)

@given(float_tensors(dtypes='float32',
                     shape=st.lists(st.integers(1, 10), min_size=2, max_size=2).map(tuple),
                     unique=True,
                     elements=floats(min_value=-100, max_value=100)))
def test_LogitCategorical(x):
    inp = torch.log_softmax(x, -1)
    dist = LogCategorical(inp)
    torch.testing.assert_allclose(dist.logits, inp)
