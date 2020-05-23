
from typing import Union, Iterable, Tuple

import functools
import gym
import numpy as np
import torch

from torforce.utils import MinMaxScaler
from .action_interface import ContinuiousActionInterface, DiscreteActionInterface, ScaledActionInterface


class StepInTerminalStateError(Exception):

    msg = 'May not call step when the enviornment is in a terminal state! (hint... call reset)'

    def __init__(self, *args, **kwargs):
        super().__init__(self.msg, *args, **kwargs)


class StatefulWrapper(gym.Wrapper):

    """This Wrapper is a base class that adds stateful functionality and pipeline interfaces to downstream wrappers.
    for the most part you would not use this wrapper directly, the only exception being a case where you want to use
    numpy arrays instead of torch Tensors in which case you could use this wrapper directly or subclass to add pipeline
    functionality.

    Args:
        env (gym.Env): gym enviornment to wrap the raw env can be accessed directly from the wrapper at the `env` attr.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._current_state = None
        self._total_reward = 0.
        self._steps = 0
        self._done = False
        self.reset()

    @property
    def name(self) ->str:
        """str: the name of the enviornment (shortcut for env.spec.id)
        """
        return self.spec.id

    @property
    def current_state(self) -> np.array:
        """this represents the current state of the env cached at every step.
        """
        return self._current_state

    @property
    def total_reward(self) ->float:
        """int: running total reward for the current episode
        """
        return self._total_reward

    @property
    def steps(self) -> int:
        """int: the number of steps in this episode that have been taken
        """
        return self._steps

    @property
    def done(self) -> bool:
        """bool: if true the episode is terminal and should be reset before the next step
        """
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

        By default the `reward_pipeline` is called for rewards on output only. This way the `total_reward`
        property will always reflect the raw reward from the enviornment regardless of your `reward_pipeline`.

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

    """Wrapper for gym envs to take some of the more annoying work out of tensorizing and untensorizing stuff while
    building agents and track a couple essential attributes.

    a TensorEnvWrapper on your gym env means

        - you get the same basic state tracking as in StateFull wrapper.
        - observations and rewards coming from the enviornment are always torch tensors
        - actions going into the enviornment as torch tensors are properly transformed back into numpy.
        - sampled actions coming out of the enviornment (via `action_space.sample` are torch tensors.)

    Args:
        env (gym.Env): gym enviornment to wrap the raw env can be accessed directly from the wrapper at the `env` attr.
        action_range (Tuple[float, float], optional): if provided action interface will be an instance of `ScaledActionInterface`
            and will rescale output actions to the range provided ( invalid for discrete action spaces obviously )
    """

    def __init__(self, env, action_range: Tuple[float, float]=None):
        super(TensorEnvWrapper, self).__init__(env)
        if self.discrete:
            self._action_interface = DiscreteActionInterface(self)
        else:
            if action_range is not None:
                self._action_interface = ScaledActionInterface(self, scaled_action_range=action_range)
            else:
                self._action_interface = ContinuiousActionInterface(self)

        self.action_space.sample = self._sample_wrapper(self.action_space.sample)

    @property
    def config(self) -> dict:
        """dict: info dict about this enviornment. should be enough to rebuild the env if needed
        """
        return dict(wrapper=f'{self.__module__}.{self.__class__.__name__}', name=self.name)

    @property
    def discrete(self) -> bool:
        """bool: if `true` the env's action space is discrete.
        """
        return str(self.env.action_space.dtype).startswith('int')

    @property
    def action_dims(self) -> int:
        """bool: if `true`  dimensions of action space... if discrete this will be the number of possible actions. If
            continious it will be the dimension of actual action space. This will generally corespond with action output
            layers in RL models.
        """
        return self._action_interface.action_dims

    @property
    def action_range(self) -> Union[tuple, None]:
        """Union[tuple, None]: in continuious action spaces this will be the min and max action range where the
            action range is finite. In discrete cases or cases where there is no finite action range this value will be
            None.
        """
        return self._action_interface.action_range

    @staticmethod
    def _tensorize(x):
        return torch.FloatTensor(x)

    def action_pipeline(self, x):
        return self._action_interface.tensor_to_action(x)

    def outgoing_action_pipeline(self, x) -> torch.Tensor:
        return self._action_interface.action_to_tensor(x)

    def observation_pipeline(self, x: np.ndarray) -> torch.Tensor:
        return self._tensorize(x)

    def reward_pipeline(self, x: np.ndarray) -> torch.Tensor:
        return self._tensorize([x])

    def _sample_wrapper(self, f):
        @functools.wraps(f)
        def inner(*args, **kwargs):
            return self.outgoing_action_pipeline(f(*args, **kwargs))
        return inner


class ScaledObservationWrapper(TensorEnvWrapper):
    """A tensor env wrapper that scales observations just scales the observation between an observation_range

    Args:
        env (gym.Env): gym enviornment to wrap the raw env can be accessed directly from the wrapper at the `env` attr.
        observation_range (tuple, optional): tuple representing new min and max values for rescaled observations.
    """

    def __init__(self, env, observation_range=(-1, 1), *args, **kwargs):
        self._observation_range = observation_range
        self.scaler = MinMaxScaler((env.observation_space.low, env.observation_space.high), self._observation_range)
        super().__init__(env, *args, **kwargs)

    @property
    def config(self) -> dict:
        """dict: config dict about this enviornment. should be enough to rebuild the env if needed
        """
        d = super().info()
        d.update(observation_range=self.observation_range)
        return d

    def observation_pipeline(self, observation):
        scaled = self.scaler.scale(observation)
        return super().observation_pipeline(scaled)
