"""Wrappers for gym learning enviornments. These wrappers were mainly written to handle a few annoyances  after a lot of
time working with gym and pytorch and looking at different implementations.

these are little issues that I think often make reading RL code more confusing which sucks:

1. It's super annoying working with mixed torch.tensors and numpy arrays.
2. Transient state passed between many functions: if we can ask the env for the current state this is simplified.
3. Lack of unified interface for simple enviornment preprocessing steps which means they often end up mixed deep in
algorithms where they become confusing and make code less extensible.

"""

from typing import Union
import gym
import numpy as np


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
