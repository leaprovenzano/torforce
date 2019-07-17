"""Policy Layers for learn a output a distribution over an action space (given some features).

Policy layers are simple output layers and generally only contain a single set of learnable weights and shold be the last layer in a policy network.
This module contiains layers for both discrete and continious action spaces.
"""
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributions import Categorical, MultivariateNormal

from torforce.distributions import UnimodalBeta, LogCategorical


class DistributionPolicyLayer(nn.Module):

    """base for all PolicyLayers. Not for use on it's own.

    Args:
        in_features (int): the number of input features
        action_dims (int): the size of the action space
    """

    Distribution = NotImplemented
    discrete = NotImplemented

    def __init__(self, in_features: int, action_dims: int):
        super().__init__()
        self.in_features = in_features
        self.action_dims = action_dims

    def _build_layer(self):
        return nn.Linear(self.in_features, self.out_features)

    def get_dist_params(self, x: torch.Tensor) -> Iterable[torch.Tensor]:
        raise NotImplementedError('subclass responsibility')

    @property
    def out_features(self) -> int:
        """int: the number of output features (alias for action_dims)
        """
        return self.action_dims

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        return self.Distribution(*self.get_dist_params(x))


class ContinuousDistributionPolicyLayer(DistributionPolicyLayer):

    """Base class for continious policies.
    """

    discrete = False


class BetaPolicyLayer(ContinuousDistributionPolicyLayer):

    """Policy layer which outputs the UnimodalBeta distribution.

    Args:
        in_features (int): the number of input features
        action_dims (int): the size of the action space

    Attributes:
        activation (nn.Softplus): softplus activation
        alpha (nn.Linear): learnable alpha parameter
        beta (nn.Linear): learnable beta parameter
        Distribution (type): distribution class UnimodalBeta or ScaledUnimodalBeta.

    Example:

        Build a ``BipedalWalker`` gym. ``BipedalWalker`` has a bounded continious action space in range ``(-1, 1)`` so
        and the Beta distribution has an output range of (0, 1) so we initilize the env with that action range so that
        actions will be rescaled on the enviorment level ( away from gradient computation etc...).
        Envs with a bounded action space are a good usecase for a `BetaPolicyLayer` since the beta distribution is bounded.

        >>> import gym
        >>> import torch
        >>> from torforce.gym_wrappers import TensorEnvWrapper
        >>> 
        >>> env = TensorEnvWrapper(gym.make('BipedalWalker-v2'), action_range=(0., 1.))
        >>> env.action_range, env.observation_space.shape[0],  env.action_dims
        (.0, 1.0), 24, 4

        Next we'll create a simple ``BetaPolicyLayer`` that just takes observations directly, in practice you will
        probably want to include more hidden layers and use ``BetaPolicyLayer`` as an output layer.

        >>> from torforce.policy.layers import BetaPolicyLayer
        >>> 
        >>> policy = BetaPolicyLayer(env.observation_space.shape[0], env.action_dims)

        >>> dist = policy(env.current_state.unsqueeze(0))
        UnimodalBeta()

    """

    Distribution = UnimodalBeta
    action_range = (0, 1)

    def __init__(self, in_features: int, action_dims: int):
        super().__init__(in_features, action_dims)
        self.alpha = self._build_layer()
        self.beta = self._build_layer()
        self.activation = nn.Softplus()

    def get_dist_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """given an input output a distribution over the action space.

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, ``self.in_features``)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: alpha and beta parameters

        """
        a, b = self.alpha(x), self.beta(x)
        a, b = self.activation(a), self.activation(b)
        return a, b

class GaussianPolicyLayer(ContinuousDistributionPolicyLayer):

    """gaussian policy layer parameterized by learnable mean and std.

    Args:
        in_features (int): the number of input features
        action_dims (int): the size of the action space
        init_std (float, optional): initial value for std parameter.

    Attributes:
        linear (nn.Linear): parameterizes the center of the gaussian.
        log_std (nn.Parameter): learnable parameter for the scale of the gaussian.

    """

    Distribution = MultivariateNormal

    def __init__(self, in_features: int, action_dims: int, init_std=1.):
        super().__init__(in_features, action_dims)
        self.linear = self._build_layer()
        self.log_std = nn.Parameter(torch.log(torch.ones(self.action_dims) * init_std))

    def get_dist_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """given an input output a distribution over the action space.
        
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, ``self.in_features``)
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: mean and std to parameterize normal distribution
        """
        mean = self.linear(x)
        cov = torch.eye(self.action_dims) * self.log_std.exp()**2
        return mean, cov



class DiscreteDistributionPolicyLayer(DistributionPolicyLayer):

    """Base class for discrete policy layers
    """

    discrete = True

class CategoricalPolicyLayer(DiscreteDistributionPolicyLayer):

    """Categorical policy layer for discrete action spaces.

    Args:
        in_features (int): the number of input features
        action_dims (int): the size of the action space

    Attributes:
        activation (nn.Softmax): the standard softmax activation.
        linear (nn.Linear): learnable layer for parameterizing the output distribution.
    """

    Distribution = Categorical

    def __init__(self, in_features: int, action_dims: int):
        super().__init__(in_features, action_dims)
        self.linear = self._build_layer()
        self.activation = nn.Softmax(dim=-1)

    def get_dist_params(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """given an input output a distribution over the action space.

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, ``self.in_features``)

        Returns:
            Tuple[torch.Tensor,]: tuple containing single torch tensor of probabilities

        """
        x = self.linear(x)
        x = self.activation(x)
        return (x,)


class LogCategoricalPolicyLayer(CategoricalPolicyLayer):

    """Categorical policy layer for discrete action spaces, the same as the CategoricalPolicyLayer but works in logspace.

    Args:
        in_features (int): the number of input features
        action_dims (int): the size of the action space

    Attributes:
        activation (nn.LogSoftmax): log softmax activation.
        linear (nn.Linear): learnable layer for parameterizing the output distribution.
    """

    Distribution = LogCategorical

    def __init__(self, in_features: int, action_dims: int):
        super().__init__(in_features, action_dims)
        self.activation = nn.LogSoftmax(dim=-1)
