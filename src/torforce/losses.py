import torch
from torch import nn


class ClippedSurrogateLoss(nn.Module):

    """Clipped surrogate loss as set out by Shulman et al. This loss is used in proximal policy optimization
    to optimize the policy network. The mean reduced negation of clipped surrogate objective as detailed in
    the paper:

    .. math::

        L^{CLIP}(\\theta) = min(r(\\theta))\hat{A}_t, clip(r(\\theta), 1-\\varepsilon, 1+\\varepsilon)\hat{A}_t)

    with an optional entropy bonus to encourage exploration.


    References:

        - Schulman, Wolski, Dhariwal, Radford and Klimov. `Proximal Policy Optimization Algorithms <https://arxiv.org/pdf/1707.06347.pdf>`_. 2017.


    Attributes:
        clip (float): used to clip the :math:`r(\\theta)` in bounds :math:`[1 + clip, 1 - clip]`. reccommended ranges are
            around .1 - .3.
        entropy_coef (float): strength of the entropy bonus good ranges will depend on the problem you're optimizing and
            the sort of distribution you're using. But in general if used this should be a very small number between (usually
            max .01), defaults to 0. (no entropy bonus)
    """

    def __init__(self, clip=.2, entropy_coef=0.):
        super().__init__()
        self.clip = clip
        self.entropy_coef = entropy_coef

    def forward(self, dist: torch.distributions.Distribution, action: torch.Tensor, advantage: torch.Tensor, old_logprob: torch.Tensor) -> torch.Tensor:
        """get the clipped surrogate loss for a distribution `dist` generated by the policy network.
        
        Args:
            dist (torch.distributions.Distribution): action distribution of the policy head being optimized
            action (torch.Tensor): the actual action taken during collection in the enviornment
            advantage (torch.Tensor): the estimated advantage of taking `action`
            old_logprob (torch.Tensor): the original log probability of taking `action` under the old policy.
        
        Returns:
            torch.Tensor
        """
        new_logprob = dist.log_prob(action)
        ratio = torch.exp(new_logprob - old_logprob)
        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantage
        err = -torch.min(clipped, unclipped).mean()
        if self.entropy_bonus:
            err = err - dist.entropy().mean() * self.entropy_bonus
        return err
