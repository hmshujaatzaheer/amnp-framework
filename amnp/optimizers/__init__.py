"""
AMNP Optimizers
===============

Optimization algorithms for neural quantum states.

Main Components:
- GASR: Geometry-Aware Stochastic Reconfiguration
"""

from .gasr import GASR, GASRState, create_gasr_optimizer
from .quantum_geometric_tensor import (
    compute_qgt,
    compute_qgt_diagonal,
    regularize_qgt,
    solve_qgt_system,
)
from .snr_estimator import (
    estimate_snr,
    compute_effective_sample_size,
    adaptive_snr_threshold,
)

__all__ = [
    "GASR",
    "GASRState",
    "create_gasr_optimizer",
    "compute_qgt",
    "compute_qgt_diagonal",
    "regularize_qgt",
    "solve_qgt_system",
    "estimate_snr",
    "compute_effective_sample_size",
    "adaptive_snr_threshold",
]
