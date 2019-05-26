from typing import Tuple, Union

import numpy as np
import torch

from torforce.utils.descriptors import MinMaxRange




class MinMaxScaler(object):

    """Min max scaler built with MinMaxRange descriptiors. MinMaxScaler is instantiated
    with two tuples, inrange and outrange which may be floats, or matching shaped numpy
    arrays or torch.Tensors or some mix of scalar float and array. MinMaxRange will
    validate these inputs and error where they are inappropriate

    Examples:

        instantiating a MinMaxScaler with inrange and outrange will give you an instance with two
        attributes `inrange` and `outrange` which are validated instances of `MinMaxRange` 
        and interface methods `scale` and `inverse_scale`:

        >>> from torforce.utils import MinMaxScaler
        >>>
        >>> scaler = MinMaxScaler((0., 1.), (-1., 1.))
        >>> print(scaler.inrange, scaler.outrange)
        (MinMaxRange : (0.0, 1.0), MinMaxRange : (-1.0, 1.0))


        Instantiation with numpy arrays also works just fine:

        >>> import numpy as np
        >>>
        >>> arrscaler = MinMaxScaler((np.zeros(3), np.ones(3)), (np.ones(3)*-1., np.ones(3)))
        >>> arrscaler.inrange, arrscaler.outrange
        (MinMaxRange : ([0. 0. 0.], [1. 1. 1.]), MinMaxRange : ([-1. -1. -1.], [1. 1. 1.]))

        so long as the array shape matches what you're trying to scale everything will be ok:

        >>> arrscaler.scale(np.random.uniform(size=(2, 3)))
        array([[ 0.55849497,  0.56310639,  0.95245284],
               [-0.00529875,  0.13534943, -0.61108798]])


        arrays  and floats can also be mixed:

        >>> scaler = MinMaxScaler((np.zeros(3), np.ones(3)), (-1., 1))
        >>> scaler.inrange, scaler.outrange
        (MinMaxRange : ([0. 0. 0.], [1. 1. 1.]), MinMaxRange : (-1.0, 1))

        torch Tensors also work just fine:

        >>> import torch
        >>>
        >>> tensorscaler =  MinMaxScaler((torch.zeros(3), torch.ones(3)), (torch.ones(3)*-1., torch.ones(3)))
        >>> tensorscaler.scale(torch.rand(2, 3))
        tensor([[ 0.5409,  0.0543,  0.8521],
                [-0.5484,  0.7508, -0.9703]])

    Attributes:
        inrange (MinMaxRange):  MinMaxRange for input
        outrange (MinMaxRange): MinMaxRange for  output

    Args:

        inrange : input data range ((float | np.array | torch.Tensor), (float | np.array | torch.Tensor))
        outrange : output data range ((float | np.array | torch.Tensor), (float | np.array | torch.Tensor))

    """

    inrange = MinMaxRange()
    outrange = MinMaxRange()

    def __init__(self, inrange: Tuple[float, float], outrange: Tuple[float, float]):
        self.inrange = inrange
        self.outrange = outrange

    @staticmethod
    def _standardize(x: Union[np.array, torch.tensor], rng: MinMaxRange) -> Union[np.array, torch.tensor]:
        return (x - rng.min) / rng.span

    def _scale(self, x: Union[np.array, torch.tensor], a: MinMaxRange, b: MinMaxRange) -> Union[np.array, torch.tensor]:
        return self._standardize(x, a) * b.span + b.min

    @torch.no_grad()
    def scale(self, x: Union[np.array, torch.tensor]):
        """scale input from range specified in `inrange` to `outrange`.

        Args:
            x (Union[np.array, torch.tensor]): input should be in `inrange`

        Example:

            >>> import torch
            >>>
            >>> scaler =  MinMaxScaler((torch.zeros(3), torch.ones(3)),
            ...                       (torch.ones(3)*-1., torch.ones(3)))
            >>>
            >>> inp = torch.Tensor([[.1, .2, .3],
            ...                     [.4, .5, .6]])
            >>>
            >>> scaler.scale(inp)
            tensor([[-0.8000, -0.6000, -0.4000],
                    [-0.2000,  0.0000,  0.2000]])

        Notes:

            - where inrange or outrange where parameterized with tensors or arrays inputs should share the last dimension
            - this method is `torch.no_grad` decorated and so does not track gradient on this operation.


        """
        return self._scale(x, self.inrange, self.outrange)

    @torch.no_grad()
    def inverse_scale(self, x: Union[np.array, torch.tensor]):
        """inverse scale a transformed input back to the original range.

        Args:
            x (Union[np.array, torch.tensor]): input, usually output of previously scaled data.


        Example:

            >>> import torch
            >>>
            >>> scaler =  MinMaxScaler((torch.zeros(3), torch.ones(3)),
            ...                       (torch.ones(3)*-1., torch.ones(3)))
            >>>
            >>> inp = torch.Tensor([[.1, .2, .3],
            ...                     [.4, .5, .6]])
            >>>
            >>> scaled = scaler.scale(inp)
            >>> scaler.inverse_scale(scaled)
            tensor([[0.1000, 0.2000, 0.3000],
                    [0.4000, 0.5000, 0.6000]])

        Notes:

            - where inrange or outrange where parameterized with tensors or arrays inputs should share the last dimension
            - this method is `torch.no_grad` decorated and so does not track gradient on this operation.

        """
        return self._scale(x, self.outrange, self.inrange)

    @torch.no_grad()
    def __call__(self, x: Union[np.array, torch.tensor]):
        return self.scale(x)
