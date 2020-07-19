import pytest

from dataclasses import dataclass

from copy import deepcopy

import torch

from torforce.advantage import gae


@dataclass
class GAECase:

    reward: torch.Tensor
    values: torch.Tensor
    bootstrap: torch.Tensor
    terminals: torch.Tensor
    expected: torch.Tensor

    def unsqueeze(self, dim=-1):
        if isinstance(self.bootstrap, torch.Tensor):
            bootstrap = self.bootstrap.unsqueeze(dim)
        else:
            bootstrap = self.bootstrap
        return self.__class__(
            reward=self.reward.unsqueeze(dim),
            values=self.values.unsqueeze(dim),
            bootstrap=bootstrap,
            terminals=self.terminals.unsqueeze(dim),
            expected=self.expected.unsqueeze(dim),
        )

    def copy_with_args(self, **kwargs):
        args = deepcopy(self.__dict__)
        args.update(kwargs)
        return self.__class__(**args)


case = GAECase(
    reward=torch.tensor([-3, 0, 0.5, 1.0]),
    values=torch.tensor([0.5, 0.6, 0.7, 0.1]),
    bootstrap=torch.tensor(0.2),
    terminals=torch.tensor([False, False, True, False]),
    expected=torch.tensor([-2.9954, -0.0951, -0.2000, 1.0980]),
)

unsqueezed_case = case.unsqueeze()
float_bootstrap = case.copy_with_args(bootstrap=0.2)
no_terminal = case.copy_with_args(
    terminals=torch.tensor([False, False, False, False]),
    expected=torch.tensor([-1.9944, 0.9692, 0.9317, 1.0980]),
)

last_terminal = case.copy_with_args(
    terminals=torch.tensor([False, False, False, True]),
    expected=torch.tensor([-2.1592, 0.7941, 0.7454, 0.9000]),
)


@pytest.mark.parametrize(
    'case,', [case, unsqueezed_case, float_bootstrap, no_terminal, last_terminal]
)
def test_gae(case):
    result = gae(case.reward, case.values, case.terminals, case.bootstrap)
    torch.testing.assert_allclose(result, case.expected)


def test_last_terminal_ignores_bootstrap():
    case = last_terminal
    with_bootstrap = gae(case.reward, case.values, case.terminals, bootstrap=case.bootstrap)
    without_bootstrap = gae(case.reward, case.values, case.terminals)
    torch.testing.assert_allclose(without_bootstrap, with_bootstrap)
    torch.testing.assert_allclose(without_bootstrap, case.expected)
