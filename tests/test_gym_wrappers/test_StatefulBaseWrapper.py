import pytest

from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, precondition

import numpy as np
import gym
from torforce.gym_wrappers import StatefulBaseWrapper, StepInTerminalStateError





def test_calling_step_on_done_raises_error():
    env = StatefulBaseWrapper(gym.make('CartPole-v1'))
    done = env.done
    while not done:
        env.step(env.action_space.sample())
        done = env.done
    with pytest.raises(StepInTerminalStateError):
        env.step(env.action_space.sample())



class StateFulBaseWrapperMachine(RuleBasedStateMachine):

    envname = 'CartPole-v1'

    def __init__(self):
        super().__init__()
        self.env = StatefulBaseWrapper(gym.make(self.envname))
        self.last_obs = None
        self.env.seed = 10
        self.total_reward = 0.
        self.n_steps = 0
        self.done = False

    @precondition(lambda self: not self.done)
    @rule()
    def take_step(self):
        action = self.env.action_space.sample()
        self.last_obs = self.env.current_state
        obs, reward, done, _ = self.env.step(action)
        self.total_reward += reward
        self.n_steps += 1
        self.done = done
        print(f'STEP : {self.env.steps} done: {self.env.done} {self.done} r : {self.env.total_reward}')

    @precondition(lambda self: self.done)
    @rule()
    def call_reset(self):
        print(f'RESET: {self.env.steps} done: {self.env.done} {self.done} r : {self.env.total_reward}')
        self.total_reward = 0.
        self.n_steps = 0
        self.last_obs = self.env.current_state
        self.env.reset()
        self.done = False

    @invariant()
    def steps_equal(self):
        assert self.n_steps == self.env.steps

    @invariant()
    def reward_equal(self):
        assert self.total_reward == self.env.total_reward

    @invariant()
    def obs_is_array(self):
        assert isinstance(self.env.current_state, np.ndarray)

    @invariant()
    def obs_is_new(self):
        assert any(self.env.current_state != self.last_obs)


TestCartpoleStateMachine = StateFulBaseWrapperMachine.TestCase
