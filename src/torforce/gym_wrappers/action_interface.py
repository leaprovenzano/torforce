from typing import Union, Iterable, Tuple
import numpy as np
import torch

from torforce.utils import MinMaxScaler
from torforce.utils.generic import all_equal, all_finite


class TensorActionInterface:

    """Base class for Action interfaces. All action interfaces should inherit from this class

    Attributes:
        action_dims (int): dimensions of action space... if discrete this will be the number of possible actions. If
            continious it will be the dimension of actual action space. This will generally corespond with action output
            layers in RL models.
        action_range (tuple | None): None by default and always None for discrete actions. Otherwise min and max action
            range.
    """

    def __init__(self, env):
        self.action_dims = self._get_action_dims(env)
        self.action_range = self._get_action_range(env)

    @property
    def output_action_range(self):
        return self.action_range

    def _get_action_range(self, env):
        return None

    def _get_action_dims(self, env):
        raise NotImplementedError('get_action_dims not implemented: subclass responsibility')

    def tensor_to_action(self, x):
        raise NotImplementedError('tensor_to_action not implemented: subclass responsibility')

    def action_to_tensor(self, x):
        raise NotImplementedError('action_to_tensor not implemented: subclass responsibility')


class DiscreteActionInterface(TensorActionInterface):

    """Interface For discrete actions. here `action_range` will always be None and `action_dims` will be the number of
    actions available in the space.

    Attributes:
        action_dims (int): dimensions of action space this will be the number of possible actions.
        action_range (None): action_range is always None for discrete action spaces
    """

    def __init__(self, env):
        super().__init__(env)

    def _get_action_dims(self, env):
        return env.action_space.n

    def tensor_to_action(self, x: torch.IntTensor) -> int:
        return x.item()

    def action_to_tensor(self, x: int) -> torch.IntTensor:
        return torch.IntTensor([x])


class ContinuiousActionInterface(TensorActionInterface):

    """Interface for continuious action spaces.

    Attributes:
        action_dims (int): dimensions of action space. This will generally corespond with action output layers in RL
            models.
        action_range (tuple | None): if the action space is finite and bounded this will be the min and max bounds
            of that action space, in the case that any features of the action space are not finite it will be None.
    """

    def __init__(self, env):
        super().__init__(env)

    def _get_action_range(self, env):
        action_range = env.action_space.low, env.action_space.high
        if all(map(all_finite, action_range)):
            return tuple(x[0] if all_equal(x) else x for x in action_range)
        return None

    def _get_action_dims(self, env):
        return env.action_space.shape[0]

    def tensor_to_action(self, x: torch.FloatTensor) -> np.ndarray:
        return np.asarray(x)

    def action_to_tensor(self, x: np.ndarray) -> torch.FloatTensor:
        return torch.FloatTensor(x)


class ScaledActionInterface(ContinuiousActionInterface):

    """Interface for continuious action spaces that rescales actions from the range provided to the enviornments range.

    Attributes:
        action_dims (int): dimensions of action space. This will generally corespond with action output layers in RL
            models.
        action_range (tuple | None): if the action space is finite and bounded this will be the min and max bounds
            of that action space, in the case that any features of the action space are not finite it will be None.
        scaled_action_range (Tuple[float, float]): scale incoming actions from this range to the enviornments range
    """

    def __init__(self, env, scaled_action_range: Tuple[float, float]):
        super().__init__(env)
        self.scaled_action_range = scaled_action_range
        self.scaler = MinMaxScaler( self.action_range, self.scaled_action_range)

    @property
    def output_action_range(self):
        return self.scaled_action_range

    def tensor_to_action(self, x: torch.FloatTensor) -> np.ndarray:
        return self.scaler(np.asarray(x))

    def action_to_tensor(self, x: np.ndarray) -> torch.FloatTensor:
        return torch.FloatTensor(self.scaler.inverse_scale(x))

