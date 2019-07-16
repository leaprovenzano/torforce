import numpy as np
import gym
import torch

from hypothesis import given, reproduce_failure
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tests.strategies.torchtensors import float_tensors

from torforce.gym_wrappers.action_interface import ScaledActionInterface


class TestScaledActionInterface:

    interface = ScaledActionInterface(gym.make('BipedalWalker-v2'), scaled_action_range=(0, 1))

    def test_output_action_range(self):
        assert self.interface.output_action_range == (0, 1)

    def test_action_range(self):
        assert self.interface.action_range == (-1, 1)

    @given(float_tensors(dtypes='float32',
                         shape=(4,),
                         unique=True,
                         elements=st.floats(min_value=0, max_value=1)))
    def test_tensor_to_action(self, input_tensor):
        scaled_action = self.interface.tensor_to_action(input_tensor)
        assert not (scaled_action == np.array(input_tensor, 'float32')).all()
        assert (scaled_action >= -1).all()
        assert (scaled_action <= 1).all()

    @given(arrays('float32',
                  shape=(4,),
                  unique=True,
                  elements=st.floats(min_value=-1, max_value=1)))
    def test_action_to_tensor(self, action_array):
        action_tensor = self.interface.action_to_tensor(action_array)
        print('action_array:', action_array)
        print('action_tensor:', action_tensor, '\n')
        assert not (np.asarray(action_tensor, 'float32') == action_array).all()
        assert (action_tensor >= 0).all()
        assert (action_tensor <= 1).all()


    
    def test_action_to_tensor_direct(self):
        action_tensor = self.interface.action_to_tensor(np.array([-1, .0, .5, 1]))
        print('action_tensor:', action_tensor, '\n')
        print('action_scaler', self.interface.scaler.inrange, self.interface.scaler.outrange)
        torch.testing.assert_allclose(action_tensor, torch.FloatTensor([0., .5, .75, 1.]))

    def test_tensor_to_action_direct(self):
        action_arr = self.interface.tensor_to_action(torch.FloatTensor([0., .5, .75, 1.]))
        print('action_arr:', action_arr, '\n')
        np.testing.assert_allclose(action_arr, np.array([-1, .0, .5, 1], 'float32'))
