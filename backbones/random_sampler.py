"""Random Sampling — simplest possible baseline.

Uniform random selection of npoint centers without replacement.
Serves as the weakest geometric baseline: if a method cannot beat
random sampling, it's worse than useless.
"""

import torch


def random_sampling(xyz, npoint):
    """Random uniform sampling without replacement.

    xyz: [B, N, 3]
    npoint: target number of points

    Returns indices [B, npoint]
    """
    B, N, _ = xyz.shape
    device = xyz.device

    if N <= npoint:
        return torch.arange(N, device=device).unsqueeze(0).expand(B, N)

    idx = torch.stack([torch.randperm(N, device=device)[:npoint] for _ in range(B)], dim=0)
    return idx
