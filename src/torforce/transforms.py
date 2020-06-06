import sys
from typing import Optional, Sequence, Union, Callable, TypeVar, Tuple

if sys.version_info >= (3, 8):
    from typing import Literal

else:
    from typing_extensions import Literal


from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
import torch

from torforce.utils.fixed_range import FixedRange


InT = TypeVar('InT')
OutT = TypeVar('OutT')

TensorableT = TypeVar(
    'TensorableT', np.ndarray, Sequence[Union[int, float, bool]], int, float, bool
)
NumT = TypeVar('NumT', np.ndarray, torch.Tensor, int, float)

Numeric = Union[np.ndarray, torch.Tensor, int, float]


class Transform(Callable[[InT], OutT]):  # type: ignore

    """ABC for transforms."""

    def __inverse__(self) -> 'Transform':
        """override this method to define behavior for transforms which support inversion

        If inversion is supported this method should return a new transform which is the inverse
        of this transform.
        """
        return NotImplemented

    @abstractmethod
    def __call__(self, x: InT) -> OutT:
        return NotImplemented


@dataclass
class Lambda(Transform):

    """Step from an arbitrary function

    Example:
        >>> from torforce.transforms import Lambda
        >>>
        >>> step = Lambda(lambda x: x+1)
        >>> step(1)
        2
    """

    func: Callable

    def __call__(self, x):
        return self.func(x)


@dataclass
class Numpy(Transform[NumT, np.ndarray]):

    """Step that makes stuff into numpy arrays

    Example:
        >>> from torforce.transforms import Numpy
        >>> import torch
        >>>
        >>> step = Numpy(dtype='float32')
        >>> step(torch.ones(3))
        array([1., 1., 1.], dtype=float32)

        >>> step([1, 2, 3])
        array([1., 2., 3.], dtype=float32)
    """

    dtype: Optional[Union[np.dtype, str]] = None

    def __call__(self, x: NumT) -> np.ndarray:
        return np.array(x, dtype=self.dtype)


@dataclass
class Tensorize(Transform[TensorableT, torch.Tensor]):

    """A Transform for turning stuff into tensors.

    Example:
        >>> from torforce.transforms import Tensorize
        >>> import numpy as np
        >>>
        >>> t = Tensorize()
        >>> x = np.array([1, 2, 3])
        >>> t(x)
        tensor([1, 2, 3])

        >>> t(2)
        tensor(2)

        Tensorize has support for the inversion operator when input_type is provided:
        >>> t = Tensorize(input_type=np.ndarray)
        >>> (~t)(t(x))
        array([1, 2, 3])

        >>> t = Tensorize(input_type=int)
        >>> (~t)(t(2))
        2

    """
    dtype: Optional[torch.dtype] = None
    input_type: Optional[Literal[np.ndarray, float, int]] = None  # type: ignore

    def __invert__(self):
        if self.input_type in (float, int):
            return Lambda(lambda x: self.input_type(x.item()))
        elif self.input_type is np.ndarray:
            return Numpy()
        # input_type is not one of declarable
        # types return NotImplemented
        return NotImplemented

    def __call__(self, x: TensorableT) -> torch.Tensor:
        return torch.tensor(x, dtype=self.dtype)


@dataclass
class Rescale(Transform[Numeric, Numeric]):

    """Invertable Transform that for rescaling tensors, arrays and other numerics.

    Example:
        >>> from torforce.transforms import Rescale
        >>>
        >>> t = Rescale(.5)
        >>> x = 10.0
        >>> t(x)
        5.0

        Rescale has support for the inversion operator:
        >>> (~t)(t(x))
        10.0
    """

    scale: Numeric

    def __post_init__(self):
        # preform checks on the scale param.
        # check that scale won't cause zero division
        # check will differ if its a tensor/array vs a scalar
        if isinstance(self.scale, (torch.Tensor, np.ndarray)):
            if any(self.scale <= 0):
                raise ValueError(f'{self.__class__.__name__} scale parameter must be > 0')
        # otherwise its a scalar... check that...
        elif isinstance(self.scale, (float, int)):
            if self.scale <= 0:
                raise ValueError(f'{self.__class__.__name__} scale parameter must be > 0')
        # otherwise the type is its not a tensor, array, float or int... it's junk...
        else:
            raise TypeError(
                f'{self.__class__.__name__} expected scale parameter to be one of types:'
                f' [{torch.Tensor, np.ndarray, float, int}] got type {type(self.scale)}'
            )

    def __invert__(self) -> 'Rescale':
        return Rescale(scale=1 / self.scale)

    def __call__(self, x: Numeric) -> Numeric:
        return x * self.scale


@dataclass
class Recenter(Transform[Numeric, Numeric]):

    """Invertable Transform that for additive recentering of tensors, arrays and other numerics.

    Example:
        >>> from torforce.transforms import Recenter
        >>> import numpy as np
        >>>
        >>> x = np.array([10., -5., 1., -2, -6])
        >>> t = Recenter(x.mean())
        >>> t(x)
        array([10.4, -4.6,  1.4, -1.6, -5.6])

        Recenter has support for the inversion operator:
        >>> (~t)(t(x))
        array([10., -5.,  1., -2., -6.])
    """

    loc: Numeric

    def __post_init__(self):
        # preform checks on the loc param.
        if not isinstance(self.loc, (float, int, torch.Tensor, np.ndarray)):
            # otherwise the type is its not a tensor, array, float or int... it's junk...
            raise TypeError(
                f'{self.__class__.__name__} expected loc parameter to be one of types:'
                f' [{torch.Tensor, np.ndarray, float, int}] got type {type(self.loc)}'
            )

    def __invert__(self) -> 'Recenter':
        return Recenter(loc=-1 * self.loc)

    def __call__(self, x: Numeric) -> Numeric:
        return x - self.loc


@dataclass
class RangeRescale(Transform):

    """Invertable Transform for scaling between specific ranges

    Example:
        >>> from torforce.transforms import Recenter
        >>> import numpy as np
        >>>
        >>> t = RangeRescale(inrange=(-1, 1), outrange=(0, 1))
        >>> t(-1)
        0.0

        Recenter has support for the inversion operator:
        >>> (~t)(0.0)
        -1.0
    """

    inrange: Union[FixedRange, Tuple[Union[float, int, torch.Tensor, np.ndarray]]]
    outrange: Union[FixedRange, Tuple[Union[float, int, torch.Tensor, np.ndarray]]]

    def __post_init__(self):
        # transform tor FixedRange if they are not already
        if not isinstance(self.inrange, FixedRange):
            self.inrange = FixedRange(*self.inrange)
        if not isinstance(self.outrange, FixedRange):
            self.outrange = FixedRange(*self.outrange)

    def __invert__(self):
        return self.__class__(self.outrange, self.inrange)

    def __call__(self, x):
        return (
            (x - self.inrange.low) / self.inrange.scale
        ) * self.outrange.scale + self.outrange.low
