import torch

import pytest

from torforce.distributions import LogCategorical, ProbCategorical, IndyBeta, IndyNormal


def logits(*shape):
    return torch.log_softmax(torch.rand(*shape), -1)


def probs(*shape):
    return torch.softmax(torch.rand(*shape), -1)


def positive(*shape):
    return torch.rand(*shape)


def normal(*shape):
    return torch.normal(0, 1, size=shape)


def assert_all_parirs_close(x, y, **kwargs):
    for xx, yy in zip(x, y):
        torch.testing.assert_allclose(xx, yy, **kwargs)


@pytest.mark.parametrize(
    'dist_cls, params,',
    [
        (LogCategorical, (logits(5, 3),)),
        (ProbCategorical, (probs(5, 3),)),
        (IndyNormal, (normal(5, 3), positive(5, 3))),
        (IndyNormal, (normal(5, 3), torch.tensor(0.5))),
        (IndyBeta, (positive(5, 3), positive(5, 3))),
    ],
)
def test_init_and_params(dist_cls, params):
    dist = dist_cls(*params)
    assert isinstance(dist, dist_cls)
    assert isinstance(dist.params, tuple)
    assert len(dist.params) == len(params)
    assert_all_parirs_close(dist.params, params)
