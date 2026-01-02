"""
Non-Hermitian Trotter-Consistent Truncation (NHTCT)
===================================================

Extends Majorana string propagation to non-Hermitian Hamiltonians.

Key innovation: Decay-bounded truncation for complex eigenvalue spectra.

For non-Hermitian H = H_Herm + iΓ:
    Truncate if: w_s(v) > S  OR  |Im[λ_v]| > Γ_max

where Γ_max = -(1/δτ) * ln(ε_Trotter)

Reference:
- D'Anna, Nys, Carrasquilla, arXiv:2511.02809 (2025)
"""

import jax
import jax.numpy as jnp
from jax import random, lax
from typing import Tuple, List, Optional, NamedTuple, Callable
from functools import partial
from dataclasses import dataclass

from .majorana_strings import (
    MajoranaString,
    create_majorana_string,
    compute_string_weight,
    multiply_strings,
    commute_with_hamiltonian,
)
from .decay_bounds import compute_gamma_max, estimate_decay_rate


@dataclass
class NHTCTConfig:
    """Configuration for NHTCT truncation."""
    max_weight: int = 6  # S parameter
    trotter_error: float = 1e-4  # ε_Trotter
    time_step: float = 0.01  # δτ
    max_decay_rate: Optional[float] = None  # Γ_max (computed if None)
    adaptive_truncation: bool = True
    
    def __post_init__(self):
        if self.max_decay_rate is None:
            self.max_decay_rate = compute_gamma_max(
                self.time_step, self.trotter_error
            )


class TruncationStats(NamedTuple):
    """Statistics from truncation."""
    n_strings_before: int
    n_strings_after: int
    n_weight_truncated: int
    n_decay_truncated: int
    max_weight_kept: int
    max_decay_kept: float


class NHTCT:
    """
    Non-Hermitian Trotter-Consistent Truncation scheme.
    
    Extends Majorana string propagation to handle non-Hermitian Hamiltonians
    with complex eigenvalues by introducing decay-bounded pruning.
    
    Parameters
    ----------
    config : NHTCTConfig
        Configuration parameters
    hamiltonian : array or callable
        Non-Hermitian Hamiltonian H = H_Herm + iΓ
    """
    
    def __init__(
        self,
        max_weight: int = 6,
        trotter_error: float = 1e-4,
        time_step: float = 0.01,
        max_decay_rate: Optional[float] = None,
    ):
        self.config = NHTCTConfig(
            max_weight=max_weight,
            trotter_error=trotter_error,
            time_step=time_step,
            max_decay_rate=max_decay_rate,
        )
        self._truncation_history: List[TruncationStats] = []
    
    def should_truncate(
        self,
        string: MajoranaString,
        eigenvalue: complex,
    ) -> Tuple[bool, str]:
        """
        Determine if a Majorana string should be truncated.
        
        Truncation criteria:
        1. Weight truncation: w_s(v) > S
        2. Decay truncation: |Im[λ_v]| > Γ_max
        
        Parameters
        ----------
        string : MajoranaString
            Majorana string to evaluate
        eigenvalue : complex
            Associated eigenvalue
        
        Returns
        -------
        truncate : bool
            Whether to truncate
        reason : str
            Reason for truncation ('none', 'weight', 'decay')
        """
        weight = compute_string_weight(string)
        decay_rate = jnp.abs(jnp.imag(eigenvalue))
        
        if weight > self.config.max_weight:
            return True, 'weight'
        
        if decay_rate > self.config.max_decay_rate:
            return True, 'decay'
        
        return False, 'none'
    
    def truncate_strings(
        self,
        strings: List[MajoranaString],
        eigenvalues: jnp.ndarray,
    ) -> Tuple[List[MajoranaString], jnp.ndarray, TruncationStats]:
        """
        Apply truncation to a set of Majorana strings.
        
        Parameters
        ----------
        strings : list of MajoranaString
            Strings to truncate
        eigenvalues : array
            Complex eigenvalues
        
        Returns
        -------
        kept_strings : list
            Strings after truncation
        kept_eigenvalues : array
            Eigenvalues of kept strings
        stats : TruncationStats
            Truncation statistics
        """
        kept_strings = []
        kept_eigenvalues = []
        n_weight_truncated = 0
        n_decay_truncated = 0
        max_weight_kept = 0
        max_decay_kept = 0.0
        
        for string, eigenvalue in zip(strings, eigenvalues):
            truncate, reason = self.should_truncate(string, eigenvalue)
            
            if truncate:
                if reason == 'weight':
                    n_weight_truncated += 1
                else:
                    n_decay_truncated += 1
            else:
                kept_strings.append(string)
                kept_eigenvalues.append(eigenvalue)
                weight = compute_string_weight(string)
                decay = jnp.abs(jnp.imag(eigenvalue))
                max_weight_kept = max(max_weight_kept, weight)
                max_decay_kept = max(max_decay_kept, decay)
        
        stats = TruncationStats(
            n_strings_before=len(strings),
            n_strings_after=len(kept_strings),
            n_weight_truncated=n_weight_truncated,
            n_decay_truncated=n_decay_truncated,
            max_weight_kept=max_weight_kept,
            max_decay_kept=max_decay_kept,
        )
        self._truncation_history.append(stats)
        
        return kept_strings, jnp.array(kept_eigenvalues), stats
    
    def evolve(
        self,
        hamiltonian: jnp.ndarray,
        initial_state: jnp.ndarray,
        t_final: float,
        observables: Optional[List[str]] = None,
        callback: Optional[Callable] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Evolve state under non-Hermitian Hamiltonian using NHTCT.
        
        Parameters
        ----------
        hamiltonian : array
            Non-Hermitian Hamiltonian matrix
        initial_state : array
            Initial state vector
        t_final : float
            Final time
        observables : list of str, optional
            Observables to compute during evolution
        callback : callable, optional
            Called at each time step
        
        Returns
        -------
        times : array
            Time points
        states : array
            State vectors at each time
        """
        dt = self.config.time_step
        n_steps = int(t_final / dt)
        
        # Decompose into Hermitian and anti-Hermitian parts
        H_herm = 0.5 * (hamiltonian + hamiltonian.conj().T)
        H_anti = 0.5 * (hamiltonian - hamiltonian.conj().T)
        gamma = -1j * H_anti  # Dissipator Γ
        
        times = jnp.linspace(0, t_final, n_steps + 1)
        states = [initial_state]
        
        psi = initial_state
        
        for step in range(n_steps):
            # Trotter step: e^{-i(H_herm + iΓ)dt} ≈ e^{-iH_herm dt} e^{-Γ dt}
            
            # Hermitian evolution
            psi = self._hermitian_step(psi, H_herm, dt)
            
            # Non-Hermitian decay
            psi = self._decay_step(psi, gamma, dt)
            
            # Normalize (for dissipative dynamics)
            psi = psi / jnp.linalg.norm(psi)
            
            states.append(psi)
            
            if callback is not None:
                callback(step, times[step + 1], psi)
        
        return times, jnp.stack(states)
    
    def _hermitian_step(
        self,
        psi: jnp.ndarray,
        H: jnp.ndarray,
        dt: float,
    ) -> jnp.ndarray:
        """Single Hermitian time step."""
        # Use Padé approximant for matrix exponential
        U = jax.scipy.linalg.expm(-1j * H * dt)
        return U @ psi
    
    def _decay_step(
        self,
        psi: jnp.ndarray,
        gamma: jnp.ndarray,
        dt: float,
    ) -> jnp.ndarray:
        """Apply decay from anti-Hermitian part."""
        D = jax.scipy.linalg.expm(-gamma * dt)
        return D @ psi
    
    @property
    def truncation_history(self) -> List[TruncationStats]:
        """Return history of truncation statistics."""
        return self._truncation_history
    
    def get_truncation_summary(self) -> dict:
        """Summarize truncation statistics."""
        if not self._truncation_history:
            return {}
        
        total_before = sum(s.n_strings_before for s in self._truncation_history)
        total_after = sum(s.n_strings_after for s in self._truncation_history)
        total_weight = sum(s.n_weight_truncated for s in self._truncation_history)
        total_decay = sum(s.n_decay_truncated for s in self._truncation_history)
        
        return {
            'total_strings_processed': total_before,
            'total_strings_kept': total_after,
            'truncation_rate': 1 - total_after / total_before,
            'weight_truncation_fraction': total_weight / (total_before - total_after + 1),
            'decay_truncation_fraction': total_decay / (total_before - total_after + 1),
        }


def create_nhtct(
    max_weight: int = 6,
    trotter_error: float = 1e-4,
    time_step: float = 0.01,
) -> NHTCT:
    """
    Factory function to create NHTCT scheme.
    
    Parameters
    ----------
    max_weight : int
        Maximum Majorana string weight S
    trotter_error : float
        Trotter error tolerance ε
    time_step : float
        Time step δτ
    
    Returns
    -------
    nhtct : NHTCT
        Configured truncation scheme
    """
    return NHTCT(
        max_weight=max_weight,
        trotter_error=trotter_error,
        time_step=time_step,
    )
