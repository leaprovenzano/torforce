import torch
from torch import nn
from torforce.activations import SoftPlusOne
from torforce.distributions import IndyBeta, IndyNormal


class BetaPolicyLayer(nn.Module):

    """Policy layer for bounded continuous outputs a  beta distribution.

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


class GaussianPolicyLayer(nn.Module):

    """Policy layer for unbounded continuous action spaces: learns a normal distribution.

    Args:
        in_features: number of input features
        action_dims: number of dims in the action space
        init_std: initial std deviation (default: 1.0)
        learned_std: boolean indicating if we should learn std deviation as a parameter

    Example:
        >>> from torforce.modules import GaussianPolicyLayer
        >>> import torch
        >>> _ = torch.manual_seed(2)
        >>>
        >>> action_dims = 4
        >>>
        >>> policy_layer = GaussianPolicyLayer(10, 4, init_std=.5, learned_std=False)
        >>> policy_layer.std
        tensor(0.5000)

        >>> hidden = torch.normal(0, 2, size=(3, 10)) # (bs, hidden)
        >>> dist = policy_layer(hidden)
        >>> dist.sample()
        tensor([[-0.9229, -2.8325, -1.3948, -0.8284],
                [ 0.2909, -0.4104, -0.6796, -0.2392],
                [-0.4066, -0.2749, -1.6538, -1.6742]])
    """
    action_range = (-float('inf'), float('inf'))

    def __init__(
        self, in_features: int, action_dims: int, init_std: float = 1.0, learned_std: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = self.action_dims = action_dims
        self.loc = nn.Linear(self.in_features, self.out_features)
        self.learned_std = learned_std
        self.log_std = torch.log(torch.tensor(init_std))
        if self.learned_std:
            self.log_std = torch.nn.Parameter(self.log_std)

    @property
    def std(self):
        return torch.exp(self.log_std)

    def set_std(self, std):
        if self.learned_std:
            raise ValueError('learned stds cannot be set')
        self.log_std = torch.log(std)

    def build_dist(self, loc: torch.Tensor, std: torch.Tensor) -> IndyNormal:
        return IndyNormal(loc, self.std)

    def forward(self, x: torch.Tensor) -> IndyNormal:
        return self.build_dist(self.loc(x), self.std)
