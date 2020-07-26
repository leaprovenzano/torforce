import pytest
import torch
import numpy as np

from torforce.utils import as_scalar


@pytest.mark.parametrize(
    'inp,', [np.random.uniform(size=(5,)), torch.rand(size=(5,)), torch.rand(size=(5, 10))]
)
def test_as_scalar_errors_on_non_unique(inp):
    expected_msg = (
        f'only {inp.__class__.__name__}s with a single unique value can be converted to scalars'
    )
    with pytest.raises(TypeError, match=expected_msg):
        as_scalar(inp)


@pytest.mark.parametrize(
    'inp,',
    [
        np.ones((5,)),
        torch.ones(size=(5,)),
        torch.ones(size=(5, 10)),
        torch.ones(size=(1,)),
        torch.tensor(1),
        np.array(1),
        1.0,
        1,
    ],
)
def test_as_scalar(inp):
    assert as_scalar(inp) == 1
