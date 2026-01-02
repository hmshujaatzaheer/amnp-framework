"""
AMNP: Adaptive Majorana-Neural Propagation Framework
=====================================================

A unified framework for simulating non-Hermitian quantum many-body dynamics
using neural quantum states.

Main Components:
- GASR: Geometry-Aware Stochastic Reconfiguration optimizer
- NHTCT: Non-Hermitian Trotter-Consistent Truncation
- TENGS: Thermofield-Extended Neural Gibbs States

Author: H M Shujaat Zaheer
Email: shujabis@gmail.com
"""

__version__ = "0.1.0"
__author__ = "H M Shujaat Zaheer"
__email__ = "shujabis@gmail.com"

from amnp.optimizers import GASR
from amnp.truncation import NHTCT
from amnp.neural_states import TENGS, RNNWavefunction, BiViTBackflow

__all__ = [
    "GASR",
    "NHTCT", 
    "TENGS",
    "RNNWavefunction",
    "BiViTBackflow",
]
