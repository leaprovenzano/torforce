import torch
from torch import nn
from torforce.activations import SoftPlusOne
from torforce.distributions import IndyBeta


class BetaPolicyLayer(nn.Module):

    """Policy layer for bounded action spaces: learns to parameterize a unimodal beta distribution.

    Args:
        in_features: number of input features
        action_dims: number of dims in the action space

    Example:
        >>> from torforce.modules import BetaPolicyLayer
        >>> import torch
        >>> _ = torch.manual_seed(2)
        >>>
        >>> action_dims = 4
        >>>
        >>> policy_layer = BetaPolicyLayer(10, 4)
        >>> hidden = torch.normal(0, 1, size=(3, 10)) # (bs, hidden)
        >>> dist = policy_layer(hidden)
        >>> dist.sample()
        tensor([[0.2111, 0.3599, 0.8813, 0.5579],
                [0.4048, 0.1518, 0.4362, 0.2096],
                [0.5033, 0.1499, 0.7499, 0.7165]])
    """
    action_range = (0.0, 1.0)

    def __init__(self, in_features: int, action_dims: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = self.action_dims = action_dims
        self.alpha = nn.Linear(self.in_features, self.out_features)
        self.beta = nn.Linear(self.in_features, self.out_features)
        self.activation = SoftPlusOne()

    def build_dist(self, alpha: torch.Tensor, beta: torch.Tensor) -> IndyBeta:
        return IndyBeta(alpha, beta)

    def forward(self, x: torch.Tensor) -> IndyBeta:
        alpha = self.activation(self.alpha(x))
        beta = self.activation(self.beta(x))
        return self.build_dist(alpha, beta)
