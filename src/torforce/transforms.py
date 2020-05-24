from typing import Optional, Sequence, Union, Callable, TypeVar
from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
import torch


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
class Tensorize(Transform[TensorableT, torch.Tensor]):

    """A Transform for turning stuff into tensors.

    Example:
        >>> from torforce.transforms import Tensorize
        >>> import numpy as np
        >>>
        >>> step = Tensorize()
        >>> step(np.array([1, 2, 3]))
        tensor([1, 2, 3])

        >>> step(2)
        tensor(2)
    """
    dtype: Optional[torch.dtype] = None

    def __call__(self, x: TensorableT) -> torch.Tensor:
        return torch.tensor(x, dtype=self.dtype)


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
        array([10.0, -5.0,  1.0, -2.0, -6.0])
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
