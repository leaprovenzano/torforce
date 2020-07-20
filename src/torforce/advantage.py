from typing import Union
import torch

from torforce.reward import discount


def _build_bootstrap_values(
    values: torch.Tensor, bootstrap: Union[float, torch.Tensor] = 0.0
) -> torch.Tensor:
    bs = torch.zeros_like(values[-1:]) + bootstrap
    return torch.cat([values, bs])


def gae(
    reward: torch.Tensor,
    values: torch.Tensor,
    terminals: torch.BoolTensor,
    bootstrap: Union[float, torch.Tensor] = 0.0,
    gamma: float = 0.99,
    lambd: float = 0.95,
):
    """generalised advantage estimate as described in `High-Dimensional Continuous Control
    Using Generalized Advantage Estimation`_ .

    Args:
        reward: undiscounted reward of shape (timesteps,) or (timesteps, n_envs)
        values: V(s_t) of shape (timesteps,) or (timesteps, n_envs)
        terminals: boolean tensor marking terminal states of shape(timesteps,) or
            (timesteps, n_envs)
        bootstrap: a bootstrap value for the last state in the sequence. Defaults to 0.0.
        gamma: gamma discount factor. Defaults to 0.99.
        lambd: lambda discount factor. Defaults to 0.95.

    .. _High-Dimensional Continuous Control Using Generalized Advantage
        Estimation: https://arxiv.org/abs/1506.02438
    """
    non_terminals = ~terminals * 1.0
    bootstrapped_values = _build_bootstrap_values(values, bootstrap)
    td_errors = reward + gamma * bootstrapped_values[1:] * non_terminals - bootstrapped_values[:-1]
    return discount(td_errors, terminals=terminals, bootstrap=0.0, gamma=gamma * lambd)
