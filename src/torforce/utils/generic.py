from typing import Iterable
import numpy as np


def all_equal(x: Iterable) -> bool:
    """return `True` if all items `x` are equal.
    
    Args:
        x (Iterable)
    
    Returns:
        bool: `True` if all items `x` are equal.
    """
    return all(x == min(x))


def all_finite(x: Iterable) -> bool:
    """check if all items in an iterable are finite
    
    Args:
        x (Iterable)
    
    Returns:
        bool: `True` if *all* items in the iterable `x` are finite
    """
    return all(np.isfinite(x))

