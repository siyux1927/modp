"""MOPD -- Multi-Teacher On-Policy Distillation.

A minimal, single-GPU, sandbox-free reproduction. Three stages that never co-reside:

    mopd rollout   student generates + verifier scores        -> rollout.parquet
    mopd teacher   one teacher scores those trajectories      -> teacher-<role>.parquet
    mopd train     fuse teachers, distil into the student     -> checkpoint
"""

from .fusion import FusedTarget, SparseLogprobs, fuse_teachers, union_support
from .loss import failure_aware_weights, jsd_loss
from .router import ROUTERS, route, routing_entropy

__version__ = "0.1.0"

__all__ = [
    "ROUTERS",
    "FusedTarget",
    "SparseLogprobs",
    "failure_aware_weights",
    "fuse_teachers",
    "jsd_loss",
    "route",
    "routing_entropy",
    "union_support",
]
