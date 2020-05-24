from typing import Optional, Sequence, Union, Callable, TypeVar
from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
import torch


InType = TypeVar('InType')
OutType = TypeVar('OutType')

Tensorable = TypeVar('Tensorable', np.ndarray, Sequence[Union[int, float, bool]], int, float, bool)
Numeric = TypeVar('Numeric', np.ndarray, torch.Tensor, int, float)


class Transform(Callable[[InType], OutType]):  # type: ignore

    """ABC for transforms."""

    @abstractmethod
    def __call__(self, x: InType) -> OutType:
        return NotImplemented


@dataclass
class Tensorize(Transform[Tensorable, torch.Tensor]):

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

    def __call__(self, x: Tensorable) -> torch.Tensor:
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
class Numpy(Transform[Numeric, np.ndarray]):

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

    def __call__(self, x: Numeric) -> np.ndarray:
        return np.array(x, dtype=self.dtype)
