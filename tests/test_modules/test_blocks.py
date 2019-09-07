from argparse import Namespace

import torch
from torch import nn

from hypothesis import given, example
from hypothesis import strategies as st

from torforce.modules.blocks import LinearBlock

from tests.strategies.nn_modules import activations
from tests.utils import all_finite


class TestLinearBlock():

    _linear_block_args = {
        'in_features': st.integers(1, 10),
        'out_features': st.integers(2, 20),
        'layer_norm': st.booleans(),
        'activation': (st.none() | activations()),
        'dropout_rate': st.floats(0., .9),
        'bias': st.booleans()
    }

    linear_block_args = st.fixed_dictionaries(_linear_block_args)
    linear_blocks = st.builds(LinearBlock, **_linear_block_args)


    @given(linear_block_args)
    def test_init(self, kwargs):
        args = Namespace(**kwargs)
        block = LinearBlock(**kwargs)

        assert block.in_features == args.in_features
        assert block.out_features == args.out_features

        if args.bias:
            assert isinstance(block.linear.bias, torch.Tensor)
            assert block.linear.bias.shape == torch.Size([args.out_features])
        else:
            assert block.linear.bias is None

        if args.layer_norm:
            assert hasattr(block, 'normalization')
            assert isinstance(block.normalization, nn.LayerNorm)
        else:
            assert hasattr(block, 'normalization') == False

        if args.dropout_rate > 0.:
            assert hasattr(block, 'dropout')
            assert isinstance(block.dropout, nn.Dropout)
            assert block.dropout.p == args.dropout_rate

        else:
            assert hasattr(block, 'dropout') == False

        assert block.activation == args.activation


    @given(linear_blocks)
    def test_forward(self, block):
        x = torch.rand((1, block.in_features))
        with torch.no_grad():
            y = block(x)
        assert torch.is_tensor(y)
        assert y.dtype == x.dtype
        assert y.shape == torch.Size((1, block.out_features))
        assert all_finite(y)

