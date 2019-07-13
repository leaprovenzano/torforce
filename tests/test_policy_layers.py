import torch
from torforce.policy_layers import *
from torforce.distributions import ScaledUnimodalBeta, UnimodalBeta


class PolicyLayerBaseSuite(object):
    default_in_features = 10
    default_action_shape = 4


class TestContinuousPolicyLayer(PolicyLayerBaseSuite):

    layer_cls = ContinuousPolicyLayer

    def test_not_discrete(self):
        assert self.layer_cls.discrete == False


class TestDiscretePolicyLayer(PolicyLayerBaseSuite):

    layer_cls = DiscretePolicyLayer

    def test_discrete(self):
        assert self.layer_cls.discrete == True


class TestBetaPolicyLayer(TestContinuousPolicyLayer):

    layer_cls = BetaPolicyLayer

    def test_init_with_action_range(self):
        layer = self.layer_cls(self.default_in_features, self.default_action_shape, action_range=(-1, 1))
        with torch.no_grad():
            dist = layer(torch.rand((1, self.default_in_features)))
        assert isinstance(dist, ScaledUnimodalBeta)

    def test_init_without_action_range(self):
        layer = self.layer_cls(self.default_in_features, self.default_action_shape, action_range=(-1, 1))
        with torch.no_grad():
            dist = layer(torch.rand((1, self.default_in_features)))
        assert isinstance(dist, UnimodalBeta)
