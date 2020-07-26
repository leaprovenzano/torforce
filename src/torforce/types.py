from typing import Union
from dataclasses import dataclass
import torch
import numpy as np

from torforce.utils import as_scalar


@dataclass
class Range:
    low: Union[float, torch.Tensor, np.ndarray] = -float('inf')
    high: Union[float, torch.Tensor, np.ndarray] = float('inf')

    def __post_init__(self):
        try:
            self.high = as_scalar(self.high)
        except TypeError:
            pass
        try:
            self.low = as_scalar(self.low)
        except TypeError:
            pass
        self.span = self.high - self.low
        self.shape = self.span.shape if hasattr(self.span, 'shape') else ()
        self._validate()

    def _validate(self):
        is_valid = self.span > 0
        if self.shape:
            is_valid = all(is_valid)
        if not is_valid:
            raise TypeError('cannot instantiate a Range where low >= high')

    def __contains__(self, value) -> bool:
        inrange = self.low <= value <= self.high
        if hasattr(inrange, '__len__'):
            return all(inrange)
        return inrange

    def is_finite(self):
        return self.span != float('inf')

    @property
    def scale(self):
        return self.span

    def transform_to(self, other: 'Range'):
        if not isinstance(other, Range):
            return NotImplemented
        from torforce.transforms import RangeRescale

        return RangeRescale(self, other)
