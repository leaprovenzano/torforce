from typing import Union
import numpy as np
import torch
from dataclasses import dataclass

Numeric = Union[float, np.ndarray, torch.Tensor]


@dataclass
class FixedRange:

    low: Numeric
    high: Numeric

    def __post_init__(self):
        valid = self.low < self.high
        if isinstance(valid, (torch.Tensor, np.ndarray)):
            valid = all(valid)
        if not valid:
            raise ValueError('low must be < high')

        self.scale = self.high - self.low
