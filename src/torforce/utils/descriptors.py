import numpy as np
import torch


class ValidationDescriptor(object):

    """Generic Basclass for descriptors that preform validation
    """

    def __init__(self, value=None):
        self.value = value

    def __get__(self, instance, owner):
        return self.value

    def __set__(self, instance, value):
        self._validate(value)
        self.value = value

    def __set_name__(self, owner, name):
        self.name = name

    def _validate(self, *values):
        raise NotImplementedError(f'the _validate method for {self.__class__.__name__} has not been implemented')

    def _raise_on_invalidation(self, msg, value=None):
        meta = f'-- recieved : {value}' if value is not None else ''
        raise ValueError(f'{self.name} : {msg}{meta}')


class MinMaxRange(ValidationDescriptor):

    """Open pseudo Descriptor for range objects... pseudo because this descriptors attributes can be accessed directly
    once bound.

    MinMaxRange is parameterized by `min` and `max` which may be numpy arrays, torch tensors, floats or ints.
    Validation is preformed on min and max to be sure the values confirm to certain constraints:

        - both min and max must contain only finite values
        - min must be < max by at least `EPSILON`
        - if min and max are arrays or matrices they must have the same shape
        - if only one of min and max are arraylike the other value may be a scaler

    Attributes:
        high (float | np.array | torch.Tensor): high end of range
        low (float | np.array | torch.Tensor): low end of range

    """

    # minimum span between high and low.
    EPSILON = 1e-4

    def __init__(self, low=None, high=None):
        self.low = low
        self.high = high

    @property
    def value(self):
        return self.low, self.high

    @property
    def span(self):
        return abs(self.high - self.low)

    @property
    def min(self):
        return self.low

    @property
    def max(self):
        return self.high

    def __set__(self, instance, rng):
        self._validate(*rng)
        self.low, self.high = rng

    def __get__(self, instance, owner):
        return self

    def __repr__(self):
        return f'{self.__class__.__name__} : ({self.low}, {self.high})'

    def __eq__(self, other):
        return self.low == other.low and self.high == other.high

    def _validate_values(self, values, attrname):
        if not np.all(np.isfinite(np.asarray(values))):
            msg = f'values {self.name}.{attrname} where are not finite (and are therefore invalid)'
            self._raise_on_invalidation(msg, values)
        return True

    def _type_and_dims_validate(self, low, high):

        if isinstance(low, type(high)):
            if isinstance(low, (np.ndarray, torch.Tensor)):
                if not low.shape == high.shape:
                    self._raise_on_invalidation('low and high values must be of the same shape',
                                                f'low= {low.shape}, high= {high.shape}')

        elif isinstance(low, (np.ndarray, torch.Tensor)):
            if not isinstance(high, (float, int)):
                self._raise_on_invalidation(
                    f'if low is of type {type(low)} high must be of same type and shape OR a scaler')
            elif isinstance(high, (np.ndarray, torch.Tensor)):
                if not isinstance(low, (float, int)):
                    self._raise_on_invalidation(
                        f'if high is of type {type(high)} low must be of same type and shape OR a scaler')

    def _validate(self, low, high):
        self._validate_values(low, 'low')
        self._validate_values(high, 'high')

        # if check dims should this be an array or tensor
        self._type_and_dims_validate(low, high)

        if np.any(np.asarray(high - low) < self.EPSILON):
            self._raise_on_invalidation(
                f'low values must be less than high values with a margin of at least {self.EPSILON}', f'low= {low}  high= {high}')

        return True
