import torch

from hypothesis import strategies as st
from hypothesis import given


from torforce.policy.distribution_layers import *
from torforce.distributions import UnimodalBeta

from tests.strategies.torchtensors import float_tensors


class PolicyLayerBaseSuite(object):
    default_in_features = 10
    default_action_shape = 4


class TestContinuousLayer(PolicyLayerBaseSuite):

    layer_cls = ContinuousDistributionLayer

    def test_not_discrete(self):
        assert self.layer_cls.discrete == False


class TestDiscreteLayer(PolicyLayerBaseSuite):

    layer_cls = DiscreteDistributionLayer

    def test_discrete(self):
        assert self.layer_cls.discrete == True


class TestBetaLayer(TestContinuousLayer):

    layer_cls = UnimodalBetaLayer

    def test_init_without_action_range(self):
        layer = self.layer_cls(self.default_in_features, self.default_action_shape)
        with torch.no_grad():
            dist = layer(torch.rand((1, self.default_in_features)))
        assert isinstance(dist, UnimodalBeta)


class TestCategoricalPolicyLayer(TestDiscreteLayer):

    layer_cls = CategoricalLayer

    @given(float_tensors(dtypes='float32',
                         shape=st.lists(st.integers(1, 10), min_size=2, max_size=2).map(tuple),
                         unique=True,
                         elements=st.floats(min_value=-100, max_value=100)))
    def test_forward(self, inp):
        layer = self.layer_cls(inp.shape[-1], self.default_action_shape)
        with torch.no_grad():
            dist = layer(inp)
        assert isinstance(dist, torch.distributions.Categorical)
        torch.testing.assert_allclose(dist.probs.sum(dim=-1), 1)


class TestLogCategoricalLayer(TestDiscreteLayer):

    layer_cls = LogCategoricalLayer

    @given(float_tensors(dtypes='float32',
                         shape=st.lists(st.integers(1, 10), min_size=2, max_size=2).map(tuple),
                         unique=True,
                         elements=st.floats(min_value=-100, max_value=100)))
    def test_forward(self, inp):
        layer = self.layer_cls(inp.shape[-1], self.default_action_shape)
        with torch.no_grad():
            dist = layer(inp)
        assert isinstance(dist, torch.distributions.Categorical)
        torch.testing.assert_allclose(dist.logits.exp().sum(dim=-1), 1)

