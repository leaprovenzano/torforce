import pytest

import gym
import torch
from torforce.gym_wrappers import TensorEnvWrapper, ScaledObservationWrapper


class ExpectedEnv(object):

    def __init__(self, name, action_dims, discrete, action_range=None):
        self.name = name
        self.action_dims = action_dims
        self.discrete = discrete
        self.action_range = action_range
        self.reward_dtype = self.observation_dtype = torch.FloatTensor

    @property
    def action_dtype(self):
        if self.discrete:
            return torch.IntTensor
        return torch.FloatTensor

    @property
    def action_shape(self):
        if self.discrete:
            return (1,)
        return (self.action_dims,)


class TensorEnvSuite:

    def test_config(self):
        config = self.env.config
        assert config == self.expected.config

    def test_name(self):
        assert self.env.name == self.expected.name

    def test_action_dims(self):
        assert self.env.action_dims == self.expected.action_dims

    def test_discrete(self):
        assert self.env.discrete == self.expected.discrete

    def test_action_range(self):
        assert self.env.action_range == self.expected.action_range

    def test_step(self):
        action = self.env.action_space.sample()
        assert action.shape == self.expected.action_shape
        assert isinstance(action, self.expected.action_dtype)

        observation, reward, _, _ = self.env.step(action)
        assert isinstance(observation, self.expected.observation_dtype)
        assert isinstance(reward, self.expected.reward_dtype)
        assert reward.shape == (1,)

    def test_reset(self):
        observation = self.env.reset()
        assert isinstance(observation, self.expected.observation_dtype)
        assert all(observation == self.env.current_state)


class TestCartPolev1(TensorEnvSuite):

    expected = ExpectedEnv('CartPole-v1', action_dims=2, discrete=True)
    expected.config = {'wrapper': 'torforce.gym_wrappers.wrappers.TensorEnvWrapper',
                       'name': expected.name}
    env = TensorEnvWrapper(gym.make(expected.name))


class TestLunarLanderContinuous(TensorEnvSuite):

    expected = ExpectedEnv('LunarLanderContinuous-v2', action_dims=2, discrete=False, action_range=(-1, 1))
    expected.config = {'wrapper': 'torforce.gym_wrappers.wrappers.TensorEnvWrapper',
                       'name': expected.name}
    env = TensorEnvWrapper(gym.make(expected.name))


class TestScaledObservationWrapper:

    """test scaled observation wrapper on atari ram enviornment"""

    expected = ExpectedEnv('Tutankham-ram-v4', action_dims=8, discrete=True)
    expected.config = {'wrapper': 'torforce.gym_wrappers.wrappers.ScaledObservationWrapper',
                       'name': expected.name,
                       'observation_range': (-1, 1)}
    env = ScaledObservationWrapper(gym.make('Tutankham-ram-v4'))

    def test_scaled_observation(self):
        print(self.env.current_state)
        assert all(self.env.current_state >= -1)
        assert all(self.env.current_state <= 1)


