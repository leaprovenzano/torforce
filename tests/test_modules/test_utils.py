from hypothesis import given, example
from hypothesis import strategies as st

from torch import nn

from torforce.modules.utils import get_output_shape


convnet = nn.Sequential(nn.Conv2d(3, 16, 5),
                        nn.ReLU(),
                        nn.Conv2d(16, 16, 5),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(16, 32, 3),
                        )


class MultiOutput(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(2, 5)

    def forward(self, x):
        return self.layer1(x), self.layer2(x)


@given(model=st.just(nn.Linear(3, 4)), input_shape=st.just((3,)), expected=st.just((4,)))
@example(model=convnet, input_shape=(3, 32, 32), expected=(32, 10, 10))
@example(model=MultiOutput(), input_shape=(2,), expected=[(4,), (5,)])
@example(model=nn.Bilinear(2, 4, 8), input_shape=[(2,), (4,)], expected=(8,))
def test_get_output_shape(model, input_shape, expected):
    result = get_output_shape(model, input_shape=input_shape)
    assert result == expected
