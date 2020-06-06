from typing import Union, List, Tuple
import torch
from torch import nn


def get_output_shape(
    model: nn.Module, input_shape: Union[Tuple[int], List[Tuple[int]]]
) -> Union[Tuple[int], List[Tuple[int]]]:
    """given a torch module and a specified get the output dimensions.

    Note:
        this function will call forward on the model with `@torch.no_grad`. If your model is \
        stateful in some other custom way it may cause issues.

    Args:
        model (nn.Module): a pytorch model or Module, should have a forward method that requires \
            nothing other than tensors of the input shapes you provde.
        input_shape (Union[tuple[int], list[tuple[int]]]): tuple of integers representing the input\
             shapes *excluding the batch dimension*, if your have multiple inputs `input_shape` \
             should be a list of tuples.

    Returns:
        Union[tuple[int], list[tuple[int]]]: a single tuple in the case of a single output, a list \
            of tuples for multiple outputs

    Examples:

        >>> from torch import nn
        >>> from torforce.utils.nn_utils import get_output_shape
        >>>
        >>> model = nn.Sequential(nn.Conv2d(3, 16, 5),
        ...                       nn.ReLU(),
        ...                       nn.Conv2d(16, 16, 5),
        ...                       nn.ReLU(),
        ...                       nn.MaxPool2d(2),
        ...                       nn.Conv2d(16, 32, 3),
        ...                       )
        >>>
        >>> get_output_shape(model, input_shape=(3, 32, 32))
        (32, 10, 10)

        >>> get_output_shape(nn.Bilinear(2, 4, 8), input_shape=[(2,), (4,)])
        (8,)

        >>> class MultiOutput(nn.Module):
        ...
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.layer1 = nn.Linear(2, 4)
        ...         self.layer2 = nn.Linear(2, 5)
        ...
        ...     def forward(self, x):
        ...         return self.layer1(x), self.layer2(x)
        >>>
        >>>
        >>> get_output_shape(MultiOutput(), input_shape=(2,))
        [(4,), (5,)]
    """
    batch_size = 2

    if isinstance(input_shape[0], tuple):
        x = [torch.rand(batch_size, *s) for s in input_shape]  # type: ignore
    else:
        x = [torch.rand(batch_size, *input_shape)]

    with torch.no_grad():
        y = model(*x)

    if isinstance(y, torch.Tensor):
        return tuple(y.shape[1:])  # type: ignore

    return [tuple(z.shape[1:]) for z in y]  # type: ignore
