import sys

if sys.version_info >= (3, 8):
    from functools import singledispatchmethod
else:
    from singledispatchmethod import singledispatchmethod

from typing import Sequence, Callable, Any


from torforce.transforms import Transform
from torforce.utils.functional import chain


def _noop(x):
    return x


class Pipeline(Sequence[Callable], Transform):

    """A pipeline is an immutable chain of transformations.

    Example:
        >>> import torch
        >>> from dataclasses import dataclass
        >>> from torforce.pipelines import Pipeline
        >>> from torforce.transforms import Tensorize, Transform
        >>>
        >>> @dataclass
        ... class AddN(Transform):
        ...     n: int
        ...
        ...     def __call__(self, x):
        ...         return x + self.n
        >>>
        >>> pipe = Pipeline(Tensorize(dtype=torch.int64), AddN(1), AddN(2))
        >>> pipe
        Pipeline(Tensorize(dtype=torch.int64), AddN(n=1), AddN(n=2))

        >>> pipe(1)
        tensor(4)

        you can join pipelines together to create a new pipeline using the addition operator:
        >>> other_pipe = Pipeline(AddN(600), AddN(62))
        >>> newpipe = pipe + other_pipe
        >>> newpipe
        Pipeline(Tensorize(dtype=torch.int64), AddN(n=1), AddN(n=2), AddN(n=600), AddN(n=62))

        >>> newpipe(1)
        tensor(666)
    """

    def __init__(self, *steps: Callable):
        self._steps = steps
        # cache chained call since pipes are immutable
        self._call = chain(*self._steps) if self._steps else _noop

    def __add__(self, other: Any):
        if isinstance(other, self.__class__):
            return self.__class__(*self._steps, *other._steps)
        return NotImplemented

    @singledispatchmethod
    def __getitem__(self, i):
        try:
            return self._steps[i]
        except TypeError:
            raise TypeError(
                f'{self.__class__.__name__} indices must be integers or slices, not {type(i)}'
            )

    @__getitem__.register(slice)
    def _(self, i: slice):
        return self.__class__(*self._steps[i])

    def __len__(self) -> int:
        return len(self._steps)

    def __call__(self, x):
        return self._call(x)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}{self._steps!r}'
