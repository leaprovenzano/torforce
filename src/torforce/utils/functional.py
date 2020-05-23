from typing import Callable
from functools import reduce, partial


def chain(func: Callable, *morefuncs: Callable) -> Callable:
    """chain together a sequence of one or more functions.

    Example:
        >>> from torforce.utils.functional import chain
        >>>
        >>> def add1(x):
        ...     return x + 1
        >>>
        >>> def add2(x):
        ...     return x + 2
        >>>
        >>> chain(add1, add2, add1)(5)
        9
    """
    return partial(reduce, lambda x, f: f(x), (func, *morefuncs))
