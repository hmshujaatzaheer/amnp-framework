"""
Decay Bounds for Non-Hermitian Truncation
=========================================

Computes decay rate bounds for NHTCT truncation.

For non-Hermitian H = H_Herm + iΓ, complex eigenvalues cause 
exponential growth/decay. We truncate strings with:

    |Im[λ_v]| > Γ_max = -(1/δτ) * ln(ε_Trotter)

This ensures truncation errors remain within Trotter accuracy.
"""

import jax.numpy as jnp
from typing import Tuple, Optional


def compute_gamma_max(
    time_step: float,
    trotter_error: float,
) -> float:
    """
    Compute maximum allowed decay rate Γ_max.
    
    Γ_max = -(1/δτ) * ln(ε_Trotter)
    
    Parameters
    ----------
    time_step : float
        Trotter time step δτ
    trotter_error : float
        Target Trotter error ε
    
    Returns
    -------
    gamma_max : float
        Maximum decay rate
    """
    if trotter_error <= 0 or trotter_error >= 1:
        raise ValueError("Trotter error must be in (0, 1)")
    
    gamma_max = -jnp.log(trotter_error) / time_step
    return float(gamma_max)


def estimate_decay_rate(
    eigenvalue: complex,
) -> float:
    """
    Extract decay rate from complex eigenvalue.
    
    For eigenvalue λ = E - iγ (with γ > 0 for decay):
        decay_rate = |Im[λ]| = |γ|
    
    Parameters
    ----------
    eigenvalue : complex
        Complex eigenvalue
    
    Returns
    -------
    decay_rate : float
        Absolute decay rate
    """
    return float(jnp.abs(jnp.imag(eigenvalue)))


def classify_eigenvalue(
    eigenvalue: complex,
    gamma_max: float,
    tolerance: float = 1e-10,
) -> str:
    """
    Classify eigenvalue behavior.
    
    Parameters
    ----------
    eigenvalue : complex
        Complex eigenvalue
    gamma_max : float
        Maximum allowed decay rate
    tolerance : float
        Tolerance for real eigenvalues
    
    Returns
    -------
    classification : str
        'stable': Real eigenvalue (oscillatory)
        'decaying': Im[λ] < 0 with |Im[λ]| ≤ Γ_max
        'growing': Im[λ] > 0 with |Im[λ]| ≤ Γ_max
        'fast_decaying': |Im[λ]| > Γ_max, decaying
        'fast_growing': |Im[λ]| > Γ_max, growing
    """
    imag_part = jnp.imag(eigenvalue)
    decay_rate = jnp.abs(imag_part)
    
    if decay_rate < tolerance:
        return 'stable'
    
    is_fast = decay_rate > gamma_max
    is_decaying = imag_part < 0
    
    if is_fast:
        return 'fast_decaying' if is_decaying else 'fast_growing'
    else:
        return 'decaying' if is_decaying else 'growing'


def compute_truncation_probability(
    decay_rate: float,
    gamma_max: float,
    sharpness: float = 10.0,
) -> float:
    """
    Compute soft truncation probability for a given decay rate.
    
    Uses sigmoid for smooth transition:
        p = σ(sharpness * (|γ| - Γ_max))
    
    Parameters
    ----------
    decay_rate : float
        Decay rate |Im[λ]|
    gamma_max : float
        Threshold Γ_max
    sharpness : float
        Sigmoid sharpness
    
    Returns
    -------
    probability : float
        Truncation probability in [0, 1]
    """
    x = sharpness * (decay_rate - gamma_max)
    return float(jax.nn.sigmoid(x))


def optimal_time_step(
    gamma_typical: float,
    trotter_error: float,
    safety_factor: float = 0.5,
) -> float:
    """
    Compute optimal time step given typical decay rates.
    
    We want Γ_max ≈ γ_typical / safety_factor, so:
        δτ = -ln(ε) * safety_factor / γ_typical
    
    Parameters
    ----------
    gamma_typical : float
        Typical decay rate in the system
    trotter_error : float
        Target Trotter error
    safety_factor : float
        Safety margin (< 1 keeps more states)
    
    Returns
    -------
    dt : float
        Recommended time step
    """
    if gamma_typical <= 0:
        # No decay -> use standard criteria
        return 0.01
    
    dt = -jnp.log(trotter_error) * safety_factor / gamma_typical
    return float(dt)


def analyze_spectrum(
    eigenvalues: jnp.ndarray,
    gamma_max: float,
) -> dict:
    """
    Analyze spectrum for truncation statistics.
    
    Parameters
    ----------
    eigenvalues : array
        Complex eigenvalues
    gamma_max : float
        Truncation threshold
    
    Returns
    -------
    stats : dict
        Spectrum statistics
    """
    decay_rates = jnp.abs(jnp.imag(eigenvalues))
    
    n_total = len(eigenvalues)
    n_stable = jnp.sum(decay_rates < 1e-10)
    n_truncated = jnp.sum(decay_rates > gamma_max)
    
    return {
        'n_total': int(n_total),
        'n_stable': int(n_stable),
        'n_kept': int(n_total - n_truncated),
        'n_truncated': int(n_truncated),
        'truncation_fraction': float(n_truncated / n_total),
        'max_decay_rate': float(jnp.max(decay_rates)),
        'mean_decay_rate': float(jnp.mean(decay_rates)),
        'spectral_gap': float(jnp.min(decay_rates[decay_rates > 0])) if jnp.any(decay_rates > 0) else 0.0,
    }
