"""
Quantum Geometric Tensor Computation
====================================

Computes the quantum geometric tensor (QGT) / Fisher information matrix
for variational quantum states.

S_kk' = ⟨∂_k log ψ* ∂_k' log ψ⟩ - ⟨∂_k log ψ*⟩⟨∂_k' log ψ⟩

This is the natural metric on the variational manifold and is used in
Stochastic Reconfiguration for natural gradient descent.
"""

import jax
import jax.numpy as jnp
from jax import vmap, jacfwd, jacrev
from typing import Callable, Tuple
from functools import partial


@partial(jax.jit, static_argnums=(2,))
def compute_qgt(
    params: jnp.ndarray,
    samples: jnp.ndarray,
    log_psi_fn: Callable,
    centered: bool = True,
    holomorphic: bool = False,
) -> jnp.ndarray:
    """
    Compute the Quantum Geometric Tensor (QGT).
    
    The QGT is defined as:
        S_kk' = ⟨O_k* O_k'⟩ - ⟨O_k*⟩⟨O_k'⟩
    
    where O_k = ∂_k log ψ are the log-derivatives.
    
    Parameters
    ----------
    params : array
        Variational parameters θ
    samples : array, shape (n_samples, n_sites)
        Monte Carlo samples from |ψ|²
    log_psi_fn : callable
        Function (params, x) -> log ψ(x; θ)
    centered : bool
        If True, compute centered QGT (subtract mean)
    holomorphic : bool
        If True, use holomorphic derivatives for complex params
    
    Returns
    -------
    S : array, shape (n_params, n_params)
        Quantum geometric tensor
    """
    n_samples = samples.shape[0]
    
    # Compute log-derivatives for all samples
    # O_k(x) = ∂_k log ψ(x; θ)
    def single_sample_derivs(x):
        def log_psi_wrapper(p):
            return log_psi_fn(p, x)
        
        if holomorphic:
            return jax.grad(log_psi_wrapper, holomorphic=True)(params)
        else:
            return jax.grad(log_psi_wrapper)(params)
    
    # Vectorize over samples
    O = vmap(single_sample_derivs)(samples)  # (n_samples, n_params)
    
    # Flatten if params is a pytree
    O_flat = jax.vmap(lambda o: jax.flatten_util.ravel_pytree(o)[0])(O)
    
    if centered:
        # Centered covariance: S = ⟨O* O⟩ - ⟨O*⟩⟨O⟩
        O_mean = jnp.mean(O_flat, axis=0)
        O_centered = O_flat - O_mean
        S = jnp.conj(O_centered.T) @ O_centered / n_samples
    else:
        # Uncentered: S = ⟨O* O⟩
        S = jnp.conj(O_flat.T) @ O_flat / n_samples
    
    return S


@partial(jax.jit, static_argnums=(2,))
def compute_qgt_diagonal(
    params: jnp.ndarray,
    samples: jnp.ndarray,
    log_psi_fn: Callable,
) -> jnp.ndarray:
    """
    Compute only the diagonal of the QGT (for memory efficiency).
    
    Parameters
    ----------
    params : array
        Variational parameters
    samples : array
        Monte Carlo samples
    log_psi_fn : callable
        Log-wavefunction
    
    Returns
    -------
    S_diag : array, shape (n_params,)
        Diagonal elements of QGT
    """
    def single_sample_derivs(x):
        def log_psi_wrapper(p):
            return log_psi_fn(p, x)
        return jax.grad(log_psi_wrapper)(params)
    
    O = vmap(single_sample_derivs)(samples)
    O_flat = jax.vmap(lambda o: jax.flatten_util.ravel_pytree(o)[0])(O)
    
    O_mean = jnp.mean(O_flat, axis=0)
    O_centered = O_flat - O_mean
    S_diag = jnp.mean(jnp.abs(O_centered)**2, axis=0)
    
    return S_diag


def regularize_qgt(
    S: jnp.ndarray,
    epsilon: float = 1e-4,
    mode: str = 'diagonal',
) -> jnp.ndarray:
    """
    Regularize the QGT for numerical stability.
    
    Parameters
    ----------
    S : array
        Quantum geometric tensor
    epsilon : float
        Regularization strength
    mode : str
        'diagonal': Add ε to diagonal
        'identity': Add εI
        'snr': SNR-based regularization
    
    Returns
    -------
    S_reg : array
        Regularized QGT
    """
    n = S.shape[0]
    
    if mode == 'diagonal':
        # Add ε * diag(S) to diagonal
        S_reg = S + epsilon * jnp.diag(jnp.diag(S))
    elif mode == 'identity':
        # Add εI
        S_reg = S + epsilon * jnp.eye(n)
    elif mode == 'snr':
        # SNR-based: ε_k = ε / (1 + S_kk)
        diag_S = jnp.diag(S)
        eps_k = epsilon / (1 + diag_S)
        S_reg = S + jnp.diag(eps_k)
    else:
        raise ValueError(f"Unknown regularization mode: {mode}")
    
    return S_reg


@partial(jax.jit, static_argnums=(2,))
def solve_qgt_system(
    S: jnp.ndarray,
    forces: jnp.ndarray,
    method: str = 'cg',
    max_iter: int = 100,
    tol: float = 1e-6,
) -> jnp.ndarray:
    """
    Solve the linear system S @ x = forces for natural gradient.
    
    Parameters
    ----------
    S : array
        Regularized QGT
    forces : array
        Gradient forces f_k = ⟨O_k* E_loc⟩ - ⟨O_k*⟩⟨E_loc⟩
    method : str
        'direct': Direct solve via Cholesky
        'cg': Conjugate gradient
    max_iter : int
        Max iterations for iterative solvers
    tol : float
        Convergence tolerance
    
    Returns
    -------
    x : array
        Solution (natural gradient direction)
    """
    if method == 'direct':
        # Direct solve assuming positive definite
        x = jax.scipy.linalg.solve(S, forces, assume_a='pos')
    elif method == 'cg':
        # Conjugate gradient
        x, _ = jax.scipy.sparse.linalg.cg(S, forces, maxiter=max_iter, tol=tol)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return x
