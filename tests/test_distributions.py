import torch

import pytest

from torforce.distributions import LogCategorical, Categorical, IndyBeta, IndyNormal


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
        (Categorical, (probs(5, 3),)),
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


class DistSuite:

    dist_cls = NotImplemented
    torch_dist_cls = NotImplemented

    @classmethod
    def build_dist(cls):
        return NotImplemented

    @classmethod
    def build_torch_dist(cls):
        return NotImplemented

    @classmethod
    def setup_class(cls):
        cls.dist = cls.build_dist()
        cls.torch_dist = cls.build_torch_dist()

    def test_shape(self):
        return NotImplemented

    def test_arg_constraints(self):
        assert self.dist_cls.arg_constraints == self.torch_dist_cls.arg_constraints

    def test_support_constraints(self):
        assert self.dist_cls.arg_constraints == self.torch_dist_cls.arg_constraints

    def test_re_init_from_params(self):
        new_dist = self.dist_cls(*self.dist.params)
        samp = new_dist.sample()
        torch.testing.assert_allclose(self.dist.log_prob(samp), new_dist.log_prob(samp))

    def test_against_torch_dist(self):
        samp = self.dist.sample()
        logprob = self.dist.log_prob(samp)
        torch_logprob = self.torch_dist.log_prob(samp)
        torch.testing.assert_allclose(logprob, torch_logprob)


class TestCategorical(DistSuite):

    dist_cls = Categorical
    torch_dist_cls = torch.distributions.Categorical
    p = probs(5, 3)
    p_is_logit = False

    @property
    def logits(self):
        return torch.log(self.p)

    @property
    def probs(self):
        return self.p

    @classmethod
    def build_dist(cls):
        return cls.dist_cls(cls.p)

    @classmethod
    def build_torch_dist(cls):
        return cls.torch_dist_cls(cls.p)

    def test_logits(self):
        torch.testing.assert_allclose(self.dist.logits, self.logits)

    def test_probs(self):
        torch.testing.assert_allclose(self.dist.probs, self.probs)

    def test_sample(self):
        sample = self.dist.sample()
        assert sample.shape == torch.Size([5])
        assert all(i in range(3) for i in sample)

    def test_shape_and_size(self):
        assert self.dist.shape == self.dist.size() == torch.Size([5, 3])

    def test_batch_ndims(self):
        assert self.dist.batch_ndims == 1

    def test_getitem(self):
        idx_dist = self.dist[:2]
        assert isinstance(idx_dist, self.dist_cls)
        assert idx_dist.shape == idx_dist.size() == torch.Size([2, 3])
        torch.testing.assert_allclose(idx_dist.logits, self.logits[:2])


class TestLogCategorical(TestCategorical):

    dist_cls = LogCategorical  # type: ignore
    torch_dist_cls = torch.distributions.Categorical
    p = logits(5, 3)

    @classmethod
    def build_torch_dist(cls):
        return cls.torch_dist_cls(logits=cls.p)

    @property
    def logits(self):
        return self.p

    @property
    def probs(self):
        return torch.exp(self.p)


class TestIndyNormal(DistSuite):

    dist_cls = IndyNormal
    torch_dist_cls = torch.distributions.Normal
    mean, std = (normal(5, 3), positive(5, 3))

    @classmethod
    def build_dist(cls):
        return cls.dist_cls(cls.mean, cls.std)

    @classmethod
    def build_torch_dist(cls):
        return torch.distributions.Independent(
            cls.torch_dist_cls(cls.mean, cls.std), reinterpreted_batch_ndims=1
        )

    def test_mean(self):
        torch.testing.assert_allclose(self.dist.mean, self.mean)

    def test_std(self):
        torch.testing.assert_allclose(self.dist.scale, self.std)

    def test_getitem(self):
        idx_dist = self.dist[:2]
        assert isinstance(idx_dist, self.dist_cls)
        assert idx_dist.shape == idx_dist.size() == torch.Size([2, 3])
        torch.testing.assert_allclose(idx_dist.mean, self.dist.mean[:2])
        torch.testing.assert_allclose(idx_dist.scale, self.dist.scale[:2])

    def test_sample(self):
        sample = self.dist.sample()
        assert sample.shape == self.dist.shape


class TestIndyBeta(DistSuite):

    dist_cls = IndyBeta
    torch_dist_cls = torch.distributions.Beta
    concentration1, concentration0 = (positive(5, 3), positive(5, 3))

    @classmethod
    def build_dist(cls):
        return cls.dist_cls(cls.concentration1, cls.concentration0)

    @classmethod
    def build_torch_dist(cls):
        return torch.distributions.Independent(
            cls.torch_dist_cls(cls.concentration1, cls.concentration0), reinterpreted_batch_ndims=1
        )

    def test_concentration0(self):
        torch.testing.assert_allclose(self.dist.concentration0, self.concentration0)

    def test_concentration1(self):
        torch.testing.assert_allclose(self.dist.concentration1, self.concentration1)

    def test_getitem(self):
        idx_dist = self.dist[:2]
        assert isinstance(idx_dist, self.dist_cls)
        assert idx_dist.shape == idx_dist.size() == torch.Size([2, 3])
        torch.testing.assert_allclose(idx_dist.concentration0, self.dist.concentration0[:2])
        torch.testing.assert_allclose(idx_dist.concentration1, self.dist.concentration1[:2])

    def test_sample(self):
        sample = self.dist.sample()
        assert sample.shape == self.dist.shape
        assert sample.min() >= 0
        assert sample.max() <= 1
