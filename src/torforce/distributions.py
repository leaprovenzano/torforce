from typing import ClassVar
import torch
from torforce.utils import classproperty


def _apply_agg(f, dists, **kwargs):
    assert all(map(lambda x: isinstance(x, type(dists[0])), dists))
    params = map(lambda tensors: f(tensors, **kwargs), zip(*map(lambda x: x.params, dists)))
    return dists[0].__class__(*params)


def cat(distributions, **kwargs):
    return _apply_agg(torch.cat, distributions, **kwargs)


def stack(distributions, dim=1):
    if dim == -1 or dim > distributions[0].batch_ndims:
        raise ValueError('cannot stack distributions along event dim')
    return _apply_agg(torch.stack, distributions, dim=dim)


class TorforceDistributionMixin:

    @property
    def shape(self) -> torch.Size:
        return self.batch_shape + self.event_shape  # type: ignore

    def size(self, dim=None):
        shape = self.shape
        if dim is not None:
            return shape[dim]
        return shape

    def __len__(self) -> int:
        return self.batch_shape[0].item()  # type: ignore

    @property
    def params(self):
        return self._natural_params

    @property
    def batch_ndims(self) -> int:
        return len(self.batch_shape)  # type: ignore

    def __repr__(self):
        parmstr = ', '.join(f'{p!r}' for p in self.params)
        return f'{self.__class__.__name__}({parmstr})'

    def _new_from_apply(self, f, *args, **kwargs):
        return self.__class__(*(f(p, *args, **kwargs) for p in self.params))

    def unsqueeze(self, dim: int):
        return self._new_from_apply(torch.Tensor.unsqueeze, dim=dim)

    def __getitem__(self, idx):
        return self._new_from_apply(torch.Tensor.__getitem__, idx)

    def unbind(self):
        if self.batch_ndims == 1:
            return [
                self.__class__(*map(lambda x: torch.unsqueeze(x, 0), p)) for p in zip(*self.params)
            ]
        return [self.__class__(*p) for p in zip(*self.params)]


class IndependentBase(torch.distributions.Independent, TorforceDistributionMixin):

    _base_dist_cls: ClassVar[torch.distributions.Distribution] = NotImplemented  # type: ignore
    _event_dim = -1

    @classproperty
    def arg_constraints(cls):  # noqa : N805
        return cls._base_dist_cls.arg_constraints

    @classproperty
    def support(cls):  # noqa : N805
        return cls._base_dist_cls.support

    def __init__(self, *args, **kwargs):
        base = self._base_dist_cls(*args, **kwargs)
        super().__init__(base, reinterpreted_batch_ndims=len(base.batch_shape[: self._event_dim]))

    @property
    def batch_ndims(self) -> int:
        return self.reinterpreted_batch_ndims

    @property
    def params(self):
        return self.base_dist._natural_params

    def __repr__(self):
        return TorforceDistributionMixin.__repr__(self)


class IndyBeta(IndependentBase):

    _base_dist_cls = torch.distributions.Beta  # type: ignore

    @property
    def concentration0(self):
        return self.base_dist.concentration0

    @property
    def concentration1(self):
        return self.base_dist.concentration1

    @property
    def params(self):
        return (self.concentration1, self.concentration0)


class IndyNormal(IndependentBase):

    _base_dist_cls = torch.distributions.Normal  # type: ignore

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale

    @property
    def params(self):
        return (self.loc, self.scale)


class Categorical(TorforceDistributionMixin, torch.distributions.Categorical):

    @property
    def params(self):
        return (self._param,)

    @property
    def shape(self) -> torch.Size:
        return self._param.shape


class LogCategorical(Categorical):

    def __init__(self, logits: torch.Tensor):
        super().__init__(logits=logits)
