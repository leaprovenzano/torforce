import torch

import pytest

from torforce.distributions import LogCategorical, Categorical, IndyBeta, IndyNormal
from torforce.distributions import stack, cat


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

    def _new_dist(self, *shape):
        return NotImplemented

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

    def test_stack(self):
        other_dist = self._new_dist(*self.dist.shape)
        stacked = stack([self.dist, other_dist], dim=1)
        assert isinstance(stacked, self.dist_cls)
        assert stacked.shape == torch.Size([self.dist.shape[0], 2, self.dist.shape[-1]])
        assert stacked.batch_ndims == 2
        return stacked, other_dist

    def test_cat(self):
        other_dist = self._new_dist(2, self.dist.shape[-1])
        catted = cat([self.dist, other_dist], dim=0)
        assert isinstance(catted, self.dist_cls)
        assert catted.shape == torch.Size([self.dist.shape[0] + 2, self.dist.shape[-1]])
        assert catted.batch_ndims == self.dist.batch_ndims
        assert (
            catted.sample().shape
            == torch.Size([2 + self.dist.shape[0]]) + self.dist.sample().shape[1:]
        )
        return catted, other_dist

    def test_unsqueeze(self):
        expected_shape = self.dist.shape[:1] + (1,) + self.dist.shape[1:]
        a = self.dist.unsqueeze(1)
        b = torch.unsqueeze(self.dist, 1)
        assert a.shape == b.shape == expected_shape
        assert all(a == b)


class TestCategorical(DistSuite):

    dist_cls = Categorical
    torch_dist_cls = torch.distributions.Categorical
    p = probs(5, 3)
    p_is_logit = False

    def _new_dist(self, *shape):
        return self.dist_cls(probs(*shape))

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

    def test_stack(self):
        stacked, other_dist = super().test_stack()
        assert stacked.sample().shape == torch.Size([self.dist.shape[0], 2])
        torch.testing.assert_allclose(
            stacked.probs, torch.stack((self.dist.probs, other_dist.probs), dim=1)
        )

    def test_cat(self):
        catted, other_dist = super().test_cat()
        torch.testing.assert_allclose(catted.probs, torch.cat((self.dist.probs, other_dist.probs)))


class TestLogCategorical(TestCategorical):

    dist_cls = LogCategorical  # type: ignore
    torch_dist_cls = torch.distributions.Categorical
    p = logits(5, 3)

    def _new_dist(self, *shape):
        return self.dist_cls(logits(*shape))

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

    def _new_dist(self, *shape):
        return self.dist_cls(normal(*shape), positive(*shape))

    def test_stack(self):
        stacked, other_dist = super().test_stack()
        assert stacked.sample().shape == torch.Size([self.dist.shape[0], 2, self.dist.shape[-1]])
        torch.testing.assert_allclose(
            stacked.mean, torch.stack((self.dist.mean, other_dist.mean), dim=1)
        )
        torch.testing.assert_allclose(
            stacked.scale, torch.stack((self.dist.scale, other_dist.scale), dim=1)
        )

    def test_cat(self):
        catted, other_dist = super().test_cat()
        torch.testing.assert_allclose(catted.mean, torch.cat((self.dist.mean, other_dist.mean)))
        torch.testing.assert_allclose(catted.scale, torch.cat((self.dist.scale, other_dist.scale)))


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

    def _new_dist(self, *shape):
        return self.dist_cls(positive(*shape), positive(*shape))

    def test_stack(self):
        stacked, other_dist = super().test_stack()
        assert stacked.sample().shape == torch.Size([self.dist.shape[0], 2, self.dist.shape[-1]])
        torch.testing.assert_allclose(
            stacked.concentration0,
            torch.stack((self.dist.concentration0, other_dist.concentration0), dim=1),
        )
        torch.testing.assert_allclose(
            stacked.concentration1,
            torch.stack((self.dist.concentration1, other_dist.concentration1), dim=1),
        )

    def test_cat(self):
        catted, other_dist = super().test_cat()
        torch.testing.assert_allclose(
            catted.concentration0, torch.cat((self.dist.concentration0, other_dist.concentration0))
        )
        torch.testing.assert_allclose(
            catted.concentration1, torch.cat((self.dist.concentration1, other_dist.concentration1))
        )
