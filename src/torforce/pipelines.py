import sys

if sys.version_info >= (3, 8):
    from functools import singledispatchmethod
else:
    from singledispatchmethod import singledispatchmethod

from typing import Sequence, Any


from torforce.transforms import Transform
from torforce.utils.functional import chain


def _noop(x):
    return x


class Pipeline(Sequence[Transform], Transform):

    """A pipeline is an immutable chain of transformations.

    Example:
        >>> import torch
        >>> import numpy as np
        >>> from dataclasses import dataclass
        >>> from torforce.pipelines import Pipeline
        >>> from torforce.transforms import Tensorize, Rescale, Recenter
        >>>
        >>> x = np.array([-1, 6.0, 5.0, -2., 9])
        >>>
        >>> pipe = Pipeline(Tensorize(input_type=np.ndarray),
        ...                 Recenter(loc=x.mean()),
        ...                 Rescale(x.std()))
        >>> pipe(x)
        tensor([-18.5845,  10.9817,   6.7580, -22.8082,  23.6530], dtype=torch.float64)

        if all the transforms in your pipe support inversion... we can also invert the whole
        pipeline using the inversion operator. This will gives us an easy way of reversing
        operations in a pipeline:
        >>> inverse_pipe = ~pipe
        >>> inverse_pipe
        Pipeline(Rescale(scale=0.23675686190518921), Recenter(loc=-3.4), Numpy(dtype=None))

        >>> inverse_pipe(pipe(x))
        array([-1.,  6.,  5., -2.,  9.])

        pipes can also be combined using the addition operator:
        >>> pointless_pipe = pipe + inverse_pipe
        >>> pointless_pipe(x)
        array([-1.,  6.,  5., -2.,  9.])

    """

    def __init__(self, *steps: Transform):
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

    def __invert__(self) -> 'Pipeline':
        inverted_transforms = []
        for t in reversed(self):
            inv = t.__invert__()
            if inv is NotImplemented:
                raise NotImplementedError(
                    'pipeline inversion is only supported when all pipeline transforms'
                    ' support inversion. inversion is not supported for pipeline'
                    f' member {t}.'
                )
            else:
                inverted_transforms.append(inv)
        return self.__class__(*inverted_transforms)
