from torch import nn

from hypothesis import strategies as st

_1d_activations = ['PReLU',
                 'Hardshrink',
                 'ELU',
                 'Threshold',
                 'LeakyReLU',
                 'LogSigmoid',
                 'ReLU6',
                 'Softsign',
                 'CELU',
                 'Tanh',
                 'RReLU',
                 'Softshrink',
                 'Hardtanh',
                 'SELU',
                 'LogSoftmax',
                 'Tanhshrink',
                 'Softmax',
                 'ReLU',
                 'Sigmoid',
                 'Softmin',
                 'Softplus']


def activations():
    activations = []
    for m in _1d_activations:
        if m[0].isupper():
            try:
                activations.append(getattr(nn.modules.activation, m)())
            except TypeError:
                pass
    return st.sampled_from(activations)
