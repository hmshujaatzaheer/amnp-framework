"""
Signal-to-Noise Ratio Estimator
===============================

Estimates the gradient signal-to-noise ratio (SNR) for adaptive 
optimization in variational Monte Carlo.

The SNR determines the reliability of gradient estimates:
    ρ = ||g||² / Var[g]

High SNR indicates reliable gradients -> use curvature (SR)
Low SNR indicates noisy gradients -> use first-order (Adam)
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional


def estimate_snr(
    gradients: jnp.ndarray,
    samples: jnp.ndarray,
    method: str = 'standard',
) -> float:
    """
    Estimate gradient signal-to-noise ratio.
    
    Parameters
    ----------
    gradients : array
        Estimated gradients (possibly from multiple samples)
    samples : array
        Monte Carlo samples used for gradient estimation
    method : str
        'standard': ||g||² / Var[g]
        'per_param': Mean over parameter-wise SNRs
        'effective': Effective sample size based
    
    Returns
    -------
    snr : float
        Estimated signal-to-noise ratio
    """
    if method == 'standard':
        signal = jnp.sum(gradients**2)
        # Estimate variance via bootstrap or jackknife
        noise = estimate_gradient_variance(gradients, samples)
        snr = signal / (noise + 1e-10)
    
    elif method == 'per_param':
        # Per-parameter SNR then average
        g_mean = gradients
        g_var = estimate_gradient_variance(gradients, samples)
        snr_per_param = g_mean**2 / (g_var + 1e-10)
        snr = jnp.mean(snr_per_param)
    
    elif method == 'effective':
        # Based on effective sample size
        n_eff = compute_effective_sample_size(samples)
        signal = jnp.sum(gradients**2)
        noise = signal / n_eff
        snr = signal / (noise + 1e-10)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return snr


def estimate_gradient_variance(
    gradients: jnp.ndarray,
    samples: jnp.ndarray,
    n_bootstrap: int = 10,
) -> float:
    """
    Estimate variance of gradient estimator via bootstrap.
    
    Parameters
    ----------
    gradients : array
        Gradient estimates
    samples : array
        Monte Carlo samples
    n_bootstrap : int
        Number of bootstrap samples
    
    Returns
    -------
    variance : float
        Estimated variance
    """
    # Simple variance estimate assuming independent samples
    # In practice, samples from MCMC are correlated
    n_samples = samples.shape[0]
    
    # Jackknife variance estimate
    variance = jnp.var(gradients) * n_samples / (n_samples - 1)
    
    return variance


def compute_effective_sample_size(
    samples: jnp.ndarray,
    max_lag: int = 100,
) -> float:
    """
    Compute effective sample size accounting for autocorrelations.
    
    N_eff = N / (1 + 2 * sum_k τ_k)
    
    where τ_k is the autocorrelation at lag k.
    
    Parameters
    ----------
    samples : array, shape (n_samples, ...)
        MCMC samples
    max_lag : int
        Maximum lag for autocorrelation
    
    Returns
    -------
    n_eff : float
        Effective sample size
    """
    n_samples = samples.shape[0]
    
    # Flatten samples for autocorrelation
    flat_samples = samples.reshape(n_samples, -1)
    
    # Compute autocorrelation function
    mean = jnp.mean(flat_samples, axis=0)
    centered = flat_samples - mean
    var = jnp.var(flat_samples, axis=0)
    
    # Integrated autocorrelation time estimate
    tau_int = 1.0
    for k in range(1, min(max_lag, n_samples // 2)):
        autocorr_k = jnp.mean(
            jnp.sum(centered[:-k] * centered[k:], axis=1)
        ) / (jnp.sum(var) * (n_samples - k))
        
        if autocorr_k < 0.05:  # Cutoff for noise
            break
        tau_int += 2 * autocorr_k
    
    n_eff = n_samples / tau_int
    return n_eff


def adaptive_snr_threshold(
    snr_history: jnp.ndarray,
    target_acceptance: float = 0.5,
) -> float:
    """
    Adaptively adjust SNR threshold based on history.
    
    Parameters
    ----------
    snr_history : array
        Historical SNR values
    target_acceptance : float
        Target fraction of updates using curvature
    
    Returns
    -------
    threshold : float
        Adjusted SNR threshold
    """
    # Set threshold at target_acceptance quantile
    threshold = jnp.percentile(snr_history, 100 * target_acceptance)
    return threshold
