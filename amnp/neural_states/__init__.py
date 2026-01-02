"""
AMNP Neural States
==================

Neural network representations for quantum many-body states.

Main Components:
- TENGS: Thermofield-Extended Neural Gibbs States
- RNNWavefunction: Autoregressive RNN ansatz
- BiViTBackflow: Bidirectional Vision Transformer with backflow
"""

from .tengs import TENGS, TENGSConfig, NonHermitianWorkOperator, create_tengs
from .rnn_wavefunction import RNNWavefunction, create_rnn_wavefunction
from .bivit_backflow import BiViTBackflow, BiViTBlock, MultiHeadAttention, GRU2D
from .jastrow import JastrowFactor, ExtendedJastrow, SpinJastrow

__all__ = [
    "TENGS",
    "TENGSConfig",
    "NonHermitianWorkOperator",
    "create_tengs",
    "RNNWavefunction",
    "create_rnn_wavefunction",
    "BiViTBackflow",
    "BiViTBlock",
    "MultiHeadAttention",
    "GRU2D",
    "JastrowFactor",
    "ExtendedJastrow",
    "SpinJastrow",
]
