from hypothesis import given, example
from hypothesis import strategies as st

import torch
from torch import nn

from torforce.modules.network_head import NetworkHead
from torforce.modules.utils import Flatten



convnet = nn.Sequential(nn.Conv2d(3, 16, 5),
                        nn.ReLU(),
                        nn.Conv2d(16, 16, 5),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(16, 32, 3),
                        Flatten()
                        )




class TestNeworkHead:

    def test_no_hidden(self):
        network_head = NetworkHead(1, 2)
        assert network_head.in_features == 1
        assert network_head.out_features == 2
        assert isinstance(network_head.output_layer, nn.Linear)
        assert network_head.in_features == 1, network_head.out_features == 2
        assert len([p for p in network_head.parameters()]) == 2

    def test_with_hidden(self):

        network_head = NetworkHead(1, 2, hidden=nn.Sequential(nn.Linear(1, 5),
                                                           nn.ReLU(),
                                                           nn.Linear(5, 10),
                                                           nn.ReLU()))

        assert network_head.in_features == 1
        assert network_head.out_features == 2
        assert isinstance(network_head.output_layer, nn.Linear)
        assert network_head.in_features == 1, network_head.out_features == 2
        assert len([p for p in network_head.parameters()]) == 6


    def test_with_Conv_hidden(self):

        network_head = NetworkHead((3, 24, 24), 64, hidden=convnet)

        assert network_head.in_features == (3, 24, 24)
        assert network_head.out_features == 64
        assert isinstance(network_head.output_layer, nn.Linear)
        assert network_head.in_features == (3, 24, 24), network_head.out_features == 64
        assert len([p for p in network_head.parameters()]) == 8
