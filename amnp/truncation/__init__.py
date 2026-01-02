"""
AMNP Truncation Schemes
=======================

Truncation methods for Majorana string propagation.

Main Components:
- NHTCT: Non-Hermitian Trotter-Consistent Truncation
"""

from .nhtct import NHTCT, NHTCTConfig, TruncationStats, create_nhtct
from .majorana_strings import (
    MajoranaString,
    create_majorana_string,
    compute_string_weight,
    compute_total_weight,
    multiply_strings,
    string_to_operator,
    majorana_operator,
    commute_with_hamiltonian,
)
from .decay_bounds import (
    compute_gamma_max,
    estimate_decay_rate,
    classify_eigenvalue,
    compute_truncation_probability,
    optimal_time_step,
    analyze_spectrum,
)

__all__ = [
    "NHTCT",
    "NHTCTConfig",
    "TruncationStats",
    "create_nhtct",
    "MajoranaString",
    "create_majorana_string",
    "compute_string_weight",
    "compute_total_weight",
    "multiply_strings",
    "string_to_operator",
    "majorana_operator",
    "commute_with_hamiltonian",
    "compute_gamma_max",
    "estimate_decay_rate",
    "classify_eigenvalue",
    "compute_truncation_probability",
    "optimal_time_step",
    "analyze_spectrum",
]
