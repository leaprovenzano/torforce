from __future__ import annotations

from typing import Union

import torch


def discount(rewards: torch.Tensor,
             terminals: torch.Tensor,
             bootstrap: Union[float, torch.Tensor] =0.,
             gamma: Union[float, torch.Tensor] =.99
             ) -> torch.Tensor:
    """Summary
    
    Args:
        rewards (torch.Tensor): a tensor
        terminals (torch.Tensor): Description
        bootstrap (Union[float, torch.Tensor], optional): Description
        gamma (Union[float, torch.Tensor], optional): Description
    
    Returns:
        torch.Tensor: Description
    """
    non_terminals = 1 - terminals
    future = bootstrap * gamma
    returns = torch.zeros_like(rewards)
    length = len(rewards)
    for i, (r, nonterm) in enumerate(zip(reversed(rewards), reversed(non_terminals))):
        future = r + future * gamma * nonterm
        returns[length - i - 1] += future
    return returns
