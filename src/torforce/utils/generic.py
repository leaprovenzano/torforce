from typing import Iterable, Callable
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


class classproperty:  # noqa: N801

    """class property decorator

    Example:

        >>> from torforce.utils import classproperty
        >>>
        >>> class Thing:
        ...
        ...     @classproperty
        ...     def num(self) -> int:
        ...         return 10
        >>>
        >>> Thing.num
        10

        >>> thing = Thing()
        >>> thing.num
        10

        >>> thing.num = 5
        >>> thing.num
        5
    """

    def __init__(self, f: Callable):
        self.f = f

    def __get__(self, inst, owner):
        if inst is not None:
            return self.f(inst)
        return self.f(owner)
