"""Abstract base class for 3D feature extractors that support PAS sampling.

All backbones (Point-MAE, PointNet++, DGCNN, etc.) must implement this interface
so that PAS can be plugged in uniformly.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class Base3DFeatureExtractor(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self._sampling_mode = None  # None = auto, 'fps', 'pas', 'rs', 'anomaly'

    def set_sampling_mode(self, mode):
        self._sampling_mode = mode

    @abstractmethod
    def forward(self, xyz, anomaly_scores=None):
        """
        xyz: [B, N, 3] point cloud coordinates, float32
        anomaly_scores: [B, N] optional, higher = more anomalous
        Returns: features [B, C, N'] or [B, N', C]
        """
        ...

    @abstractmethod
    def output_dim(self):
        """Feature dimension produced by this backbone."""
        ...
