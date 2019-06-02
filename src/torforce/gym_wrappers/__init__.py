"""Wrappers for gym learning enviornments. These wrappers were mainly written to handle a few annoyances  after a lot of
time working with gym and pytorch and looking at different implementations.

these are little issues that I think often make reading RL code more confusing which sucks:

1. It's super annoying working with mixed torch.tensors and numpy arrays.
2. Transient state passed between many functions: if we can ask the env for the current state this is simplified.
3. Lack of unified interface for simple enviornment preprocessing steps which means they often end up mixed deep in algorithms where they become confusing and make code less extensible.

"""

from .wrappers import TensorEnvWrapper, ScaledObservationWrapper
