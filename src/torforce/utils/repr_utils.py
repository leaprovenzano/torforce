from typing import Dict, Any


def keyword_format(**kwargs: Dict[str, Any]) -> str:
    """formats kwargs as comma joined key=value pairs.

    Example:
        >>> from torforce.utils.repr_utils import keyword_format
        >>>
        >>> keyword_format(a=1, b=2, c=3)
        'a=1, b=2, c=3'
    """
    return ', '.join([f'{k}={v}' for k, v in kwargs.items()])


def simple_repr(inst, **kwargs) -> str:
    return f'{inst.__class__.__name__}({keyword_format(**kwargs)})'
