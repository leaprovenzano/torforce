"""Wrappers for gym learning enviornments. These wrappers were mainly written to handle a few annoyances  after a lot of
time working with gym and pytorch and looking at different implementations.

these are little issues that I think often make reading RL code more confusing which sucks:

1. It's super annoying working with mixed torch.tensors and numpy arrays.
2. Transient state passed between many functions: if we can ask the env for the current state this is simplified.
3. Lack of unified interface for simple enviornment preprocessing steps which means they often end up mixed deep in
algorithms where they become confusing and make code less extensible.

"""
import functools
from typing import Union, Iterable
import gym
import numpy as np
import torch

from torforce.utils import MinMaxScaler


class StepInTerminalStateError(Exception):

    msg = 'May not call step when the enviornment is in a terminal state! (hint... call reset)'

    def __init__(self, *args, **kwargs):
        super().__init__(self.msg, *args, **kwargs)


def is_reducable(x: Iterable) -> bool:
    return all(x == x[0])


def is_finite(x: Iterable) -> bool:
    return all(np.isfinite(x))


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
        self.action_dims = self.get_action_dims(env)
        self.action_range = self.get_action_range(env)

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
        if all(map(is_finite, action_range)):
            return tuple(x[0] if is_reducable(x) else x for x in action_range)
        return None

    def _get_action_dims(self, env):
        return env.action_space.shape[0]

    def tensor_to_action(self, x: torch.FloatTensor) -> np.ndarray:
        return np.asarray(x)

    def action_to_tensor(self, x: np.ndarray) -> torch.FloatTensor:
        return torch.FloatTensor(x)


class StatefulWrapper(gym.Wrapper):

    """This Wrapper is a base class that adds stateful functionality and pipeline interfaces to downstream wrappers.
    for the most part you would not use this wrapper directly, the only exception being a case where you want to use
    numpy arrays instead of torch Tensors in which case you could use this wrapper directly or subclass to add pipeline
    functionality.

    Args:
        env (gym.Env): gym enviornment to wrap the raw env can be accessed directly from the wrapper at the `env` attr.

    Attributes:
        current_state (np.array): this represents the current state of the env and so is the most recent observation
            made in the enviornment at any given time. This attribute is updated by both the reset and step methods.
        done (bool): tracks terminal state, if true the episode in this enviornment is complete and needs to be reset.
        steps (int): the total number of steps taken in the current episode
        total_reward (float): the total reward for the current episode.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._current_state = None
        self._total_reward = 0.
        self._steps = 0
        self._done = False
        self.reset()

    @property
    def name(self):
        return self.spec.id
    

    @property
    def current_state(self):
        return self._current_state

    @property
    def total_reward(self):
        return self._total_reward

    @property
    def steps(self):
        return self._steps

    @property
    def done(self):
        return self._done

    def observation_pipeline(self, x: np.array) -> np.array:
        """handle observations from the enviornment before outputting.

        This is a no-op by default. override this method when you want to add observation processing steps in a
        subclass.

        Args:
            x (np.array): observation

        Returns:
            np.array: observation
        """
        return x

    def reward_pipeline(self, x: float) -> float:
        """handle rewards from the enviornment before outputting.

        This is a no-op by default. override this method when you want to add reward processing steps in a
        subclass.

        Note:
            by default this step will be called for rewards on output only. This way the total_reward attr will always
            reflect the raw reward from the enviornment regardless of your reward_pipeline.

        Args:
            x (np.array): reward

        Returns:
            np.array: reward
        """
        return x

    def action_pipeline(self, x: Union[np.array, int]) -> Union[np.array, int]:
        """handle incoming actions passed to enviornment, it will be called on actions input to the `step` method.

        This is a no-op by default. override this method when you want to add action processing steps in a subclass.

        Args:
            x (Union[np.array, int]): action

        Returns:
            Union[np.array, int]: action
        """
        return x

    def reset(self, **kwargs):
        self._current_state = self.observation_pipeline(self.env.reset(**kwargs))
        self._total_reward = 0.
        self._done = False
        self._steps = 0
        return self._current_state

    def step(self, action):
        """take a step in the enviornment by calling step in the unrapped env with an action that has been preprocessed
        through the `action_pipeline` method. Since we're stateful we'll cache new state as a result of taking the step
        in `current_state` and add the reward recieved to our `total_reward`, if the resulting state is terminal the
        `done` attr will also be set to True.

        Args:
            action : action will be further passed to `action_pipeline` before the unwrapped env sees it.

        Returns:
            state, reward, done, info

        Raises:
            StepInTerminalStateError: where an agent has tried to step in a terminal state raise this exception.
        """
        if self.done:
            raise StepInTerminalStateError()
        observation, reward, done, info = self.env.step(self.action_pipeline(action))
        self._current_state = self.observation_pipeline(observation)
        self._steps += 1
        self._done = done
        self._total_reward += reward
        return self.current_state, self.reward_pipeline(reward), done, info


class TensorEnvWrapper(StatefulWrapper):

    """Summary

    Args:
        env (gym.Env): gym enviornment to wrap the raw env can be accessed directly from the wrapper at the `env` attr.
    """
    
    def __init__(self, env):
        super(TensorEnvWrapper, self).__init__(env)

        if self.discrete:
            self._action_interface = DiscreteActionInterface(self)
        else:
            self._action_interface = ContinuiousActionInterface(self)

        self.action_space.sample = self._sample_wrapper(self.action_space.sample)

    @property
    def discrete(self):
        return str(self.env.action_space.dtype).startswith('int')

    @property
    def action_dims(self):
        return self._action_interface.action_dims

    @property
    def action_range(self):
        return self._action_interface.action_range

    @staticmethod
    def tensorize(x):
        return torch.FloatTensor(x)

    def action_pipeline(self, x):
        return self._action_interface.tensor_to_action(x)

    def outgoing_action_pipeline(self, x):
        return self._action_interface.action_to_tensor(x)

    def observation_pipeline(self, x):
        return self.tensorize(x)

    def reward_pipeline(self, x):
        return self.tensorize([x])

    def _sample_wrapper(self, f):
        @functools.wraps(f)
        def inner(*args, **kwargs):
            return self.outgoing_action_pipeline(f(*args, **kwargs))
        return inner


class ScaledObservationWrapper(TensorEnvWrapper):
    """ObservationWrapper for openai gym's atari  RAM envs. just scales the observation between an observation_range
    """

    def __init__(self, env, observation_range=(-1, 1)):
        self.scaler = MinMaxScaler((env.observation_space.low, env.observation_space.high), observation_range)
        super().__init__(env)
        

    def observation_pipeline(self, observation):
        scaled = self.scaler.scale(observation)
        return super().observation_pipeline(scaled)
