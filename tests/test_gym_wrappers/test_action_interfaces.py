import numpy as np
import gym
import torch

from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tests.strategies.torchtensors import float_tensors

from torforce.gym_wrappers.action_interface import ScaledActionInterface


class TestScaledActionInterface:

    env = gym.make('BipedalWalker-v2')

    @given(float_tensors(dtypes='float32',
                         shape=(4,),
                         unique=True,
                         elements=st.floats(min_value=0, max_value=1)))
    def test_tensor_to_action(self, input_tensor):
        interface = ScaledActionInterface(self.env, (0, 1))
        scaled_action = interface.tensor_to_action(input_tensor)
        assert not (scaled_action == np.array(input_tensor, 'float32')).all()
        assert (scaled_action >= -1).all()
        assert (scaled_action <= 1).all()

    @given(arrays('float32',
                  shape=(4,),
                  unique=True,
                  elements=st.floats(min_value=-1, max_value=1)))
    def test_action_to_tensor(self, action_array):
        interface = ScaledActionInterface(self.env, (0, 1))
        action_tensor = interface.action_to_tensor(action_array)
        assert not (np.asarray(action_tensor, 'float32') == action_array).all()
        assert (action_tensor >= 0).all()
        assert (action_tensor <= 1).all()
