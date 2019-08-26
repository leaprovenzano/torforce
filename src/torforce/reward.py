from __future__ import annotations

from typing import Union

import torch





def discount(rewards: torch.Tensor,
             terminals: torch.Tensor,
             bootstrap: Union[float, torch.Tensor] = 0.,
             gamma: Union[float, torch.Tensor] = .99) -> torch.Tensor:
    """simple discounted future reward parameterized by gamma.

    Args:
        rewards (torch.Tensor): tensor containing rewards to be discounted -- tensor dims should be either (timesteps,)
            or (timesteps, agents) in the case of multiple parallel agents or enviornments.
        terminals (torch.Tensor): tensor where terminal steps have a value of 1. and non-terminal steps have a value of 
            0 -- must be the same shape as rewards.
        bootstrap (Union[float, torch.Tensor], optional): Description
        gamma (Union[float, torch.Tensor], optional): Description

    Returns:
        torch.Tensor: Description
    """
    
    non_terminals = 1 - terminals.type_as(rewards)
    future = bootstrap
    returns = torch.zeros_like(rewards)
    for i in range(len(returns) - 1, -1, -1):
        returns[i] = rewards[i] + future * gamma * non_terminals[i]
        future = returns[i]
    return returns
