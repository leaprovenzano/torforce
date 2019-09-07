import torch


def all_finite(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all()
