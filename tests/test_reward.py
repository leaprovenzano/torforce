import pytest

import torch

from hypothesis import given, example
from hypothesis import strategies as st

from tests.strategies.torchtensors import float_tensors, tensors

from torforce.reward import discount


def get_groups(mask):
    groups = mask.flip(0).cumsum(0).flip(0)
    return abs(groups - groups.max())


def group_split(x, mask, return_mask=False):
    if len(mask.shape) == 1:
        groups = get_groups(mask)

    elif len(mask.shape) == 2:
        groups = torch.zeros_like(mask)
        max_grp = 0
        for i in range(mask.shape[-1]):
            g = max_grp + get_groups(mask[:, i])
            groups[:, i] = g
            max_grp = g.max() + 1

    if return_mask:
        return [(x[groups == i], mask[groups == i]) for i in range(groups.min(), groups.max() + 1)]
    return [x[groups == i] for i in range(groups.min(), groups.max() + 1)]


@st.composite
def discount_case(draw, rewards):
    reward = draw(rewards)
    shape = tuple(reward.shape)
    terminals = draw(tensors('uint8', shape, elements=st.booleans()))
    return reward, terminals


class TestDiscount:

    basic_reward_strat = float_tensors(min_dims=1,
                                       max_dims=2,
                                       min_side=1,
                                       max_side=10,
                                       elements=st.floats(-100, 100, width=32))
    ones = float_tensors(min_dims=1,
                         max_dims=2,
                         min_side=1,
                         max_side=10,
                         elements=st.floats(1, 1, width=32))

    @given(basic_reward_strat)
    def test_simple_gamma_one(self, inp):
        terms = torch.zeros_like(inp)
        terms[-1] = 1.
        expected = inp.flip(0).cumsum(0).flip(0)
        result = discount(inp, terminals=terms, gamma=1.)

        torch.testing.assert_allclose(result, expected, rtol=0.0001, atol=0.0001)

    @given(basic_reward_strat)
    def test_simple_gamma_zero(self, inp):
        terms = torch.zeros_like(inp)
        terms[-1] = 1.
        result = discount(inp, terminals=terms, gamma=0.)

        torch.testing.assert_allclose(result, inp, rtol=0.0001, atol=0.0001)

    @given(discount_case(ones))
    def test_mult_term_ones(self, inp):
        gamma = .99
        rewards, terms = inp
        result = discount(rewards, terms, gamma=gamma, bootstrap=1)
        torch.testing.assert_allclose(result[terms], rewards[terms])

        if not (terms == 1).all():
            for grp, msk in group_split(result, mask=terms, return_mask=True):

                expected = 1

                if msk[-1].item() == 0:
                    expected += 1 * gamma

                for v in grp.flip(0).tolist():
                    pytest.approx(v, .00001) == expected
                    expected += 1 + (expected * gamma)
