"""
Geometry-Aware Stochastic Reconfiguration (GASR) Optimizer
==========================================================

Addresses the optimizer incompatibility identified in Hibat-Allah et al. (2025):
"Stochastic Reconfiguration is not as effective as Adam optimizer when applied 
to RNN wave functions"

The GASR optimizer adaptively interpolates between curvature-aware (SR) and 
first-order (Adam) updates based on gradient signal-to-noise ratio.

Reference:
- Hibat-Allah et al., Commun. Phys. 8, 308 (2025)
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Optional, NamedTuple, Callable
from functools import partial
import flax.linen as nn
from flax.core import freeze, unfreeze

from .quantum_geometric_tensor import compute_qgt
from .snr_estimator import estimate_snr


class GASRState(NamedTuple):
    """State for GASR optimizer."""
    step: int
    alpha: float  # Interpolation parameter
    m: jnp.ndarray  # First moment (Adam)
    v: jnp.ndarray  # Second moment (Adam)
    snr_ema: float  # Exponential moving average of SNR


class GASR:
    """
    Geometry-Aware Stochastic Reconfiguration Optimizer.
    
    Adaptively interpolates between Stochastic Reconfiguration (curvature-aware)
    and Adam (first-order) based on the gradient signal-to-noise ratio.
    
    The update rule uses an interpolated metric:
        G_GASR = (1 - α)S + α J^T J + λI
    
    where:
        - S is the quantum geometric tensor (Fisher information)
        - J is the Jacobian of log-amplitudes
        - α = σ(log ρ - τ) adapts based on SNR ρ
        - λ is regularization
    
    Parameters
    ----------
    learning_rate : float
        Base learning rate η
    snr_threshold : float
        Threshold τ for SNR-based adaptation
    regularization : float
        Diagonal regularization λ
    beta1 : float
        Adam first moment decay
    beta2 : float
        Adam second moment decay
    snr_ema_decay : float
        Decay rate for SNR exponential moving average
    min_alpha : float
        Minimum value of α (ensures some curvature information)
    max_alpha : float
        Maximum value of α (ensures some first-order behavior)
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        snr_threshold: float = 10.0,
        regularization: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        snr_ema_decay: float = 0.99,
        min_alpha: float = 0.01,
        max_alpha: float = 0.99,
    ):
        self.lr = learning_rate
        self.tau = snr_threshold
        self.lambda_reg = regularization
        self.beta1 = beta1
        self.beta2 = beta2
        self.snr_ema_decay = snr_ema_decay
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
    
    def init(self, params: jnp.ndarray) -> GASRState:
        """Initialize optimizer state."""
        flat_params = jax.flatten_util.ravel_pytree(params)[0]
        return GASRState(
            step=0,
            alpha=0.5,  # Start balanced
            m=jnp.zeros_like(flat_params),
            v=jnp.zeros_like(flat_params),
            snr_ema=self.tau,
        )
    
    def _compute_alpha(self, snr: float) -> float:
        """Compute adaptive interpolation parameter."""
        # α = σ(log ρ - τ) where σ is sigmoid
        log_snr = jnp.log(snr + 1e-10)
        alpha = jax.nn.sigmoid(log_snr - jnp.log(self.tau))
        return jnp.clip(alpha, self.min_alpha, self.max_alpha)
    
    @partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: GASRState,
        params: jnp.ndarray,
        grads: jnp.ndarray,
        samples: jnp.ndarray,
        log_psi_fn: Callable,
    ) -> Tuple[GASRState, jnp.ndarray]:
        """
        Perform one GASR update step.
        
        Parameters
        ----------
        state : GASRState
            Current optimizer state
        params : array
            Model parameters
        grads : array
            Energy gradients (forces)
        samples : array
            Monte Carlo samples from |ψ|²
        log_psi_fn : callable
            Function computing log ψ(x; θ)
        
        Returns
        -------
        new_state : GASRState
            Updated optimizer state
        new_params : array
            Updated parameters
        """
        flat_grads, unravel_fn = jax.flatten_util.ravel_pytree(grads)
        flat_params, _ = jax.flatten_util.ravel_pytree(params)
        n_params = len(flat_params)
        
        # Estimate signal-to-noise ratio
        snr = estimate_snr(flat_grads, samples)
        snr_ema = self.snr_ema_decay * state.snr_ema + (1 - self.snr_ema_decay) * snr
        
        # Compute adaptive alpha
        alpha = self._compute_alpha(snr_ema)
        
        # Compute quantum geometric tensor S
        S = compute_qgt(params, samples, log_psi_fn)
        
        # Compute Jacobian term J^T J (approximated via outer product of gradients)
        JtJ = jnp.outer(flat_grads, flat_grads) / (jnp.linalg.norm(flat_grads)**2 + 1e-10)
        
        # Construct interpolated metric G_GASR
        G = (1 - alpha) * S + alpha * JtJ + self.lambda_reg * jnp.eye(n_params)
        
        # Solve for natural gradient: G_GASR @ δθ = -g
        # Use conjugate gradient for efficiency
        delta_params = jax.scipy.linalg.solve(G, -flat_grads, assume_a='pos')
        
        # Adam moment updates (for momentum)
        m = self.beta1 * state.m + (1 - self.beta1) * delta_params
        v = self.beta2 * state.v + (1 - self.beta2) * delta_params**2
        
        # Bias correction
        step = state.step + 1
        m_hat = m / (1 - self.beta1**step)
        v_hat = v / (1 - self.beta2**step)
        
        # Final update with Adam-style normalization
        update = self.lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
        new_flat_params = flat_params + update
        new_params = unravel_fn(new_flat_params)
        
        new_state = GASRState(
            step=step,
            alpha=alpha,
            m=m,
            v=v,
            snr_ema=snr_ema,
        )
        
        return new_state, new_params
    
    def __repr__(self) -> str:
        return (
            f"GASR(lr={self.lr}, τ={self.tau}, λ={self.lambda_reg}, "
            f"β1={self.beta1}, β2={self.beta2})"
        )


def create_gasr_optimizer(
    learning_rate: float = 1e-3,
    snr_threshold: float = 10.0,
    **kwargs
) -> GASR:
    """
    Factory function to create GASR optimizer.
    
    Parameters
    ----------
    learning_rate : float
        Learning rate
    snr_threshold : float
        SNR threshold for alpha adaptation
    **kwargs
        Additional arguments passed to GASR
    
    Returns
    -------
    optimizer : GASR
        Configured GASR optimizer
    """
    return GASR(
        learning_rate=learning_rate,
        snr_threshold=snr_threshold,
        **kwargs
    )
