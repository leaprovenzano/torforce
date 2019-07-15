import numpy as np
import gym
import pytest
import torch


from torforce.gym_wrappers.action_interface import ScaledActionInterface

@pytest.fixture(scope='class')
def bipedalwalker_env():
    env = gym.make('BipedalWalker-v2')
    yield env
    env.close()



@pytest.mark.usefixtures("bipedalwalker_env")
class TestScaledActionInterface:


    def test_tensor_to_action(self, bipedalwalker_env):
        interface = ScaledActionInterface(bipedalwalker_env, (0, 1))
        scaled_action = interface.tensor_to_action(torch.Tensor([.0, .5, 1., 1.]))
        np.testing.assert_almost_equal(scaled_action, np.array([-1, 0., 1., 1.]))

    def test_action_to_tensor(self, bipedalwalker_env):
        interface = ScaledActionInterface(bipedalwalker_env, (0, 1))
        scaled_action = interface.action_to_tensor(np.array([-1, 0., 1., 1.]))
        torch.testing.assert_allclose(scaled_action, torch.Tensor([.0, .5, 1., 1.]))
