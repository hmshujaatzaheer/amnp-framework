"""
Observables for Quantum Many-Body Systems
=========================================

Compute physical observables:
- Energy and variance
- Correlation functions
- Structure factors
- Entanglement entropy
"""

import jax
import jax.numpy as jnp
from jax import vmap
from typing import Tuple, Optional, Callable


def compute_energy(
    samples: jnp.ndarray,
    log_psi_fn: Callable,
    hamiltonian: jnp.ndarray,
) -> Tuple[float, float]:
    """
    Compute variational energy estimate.
    
    E = ⟨H⟩ = ∑_x |ψ(x)|² E_loc(x)
    
    where E_loc(x) = ⟨x|H|ψ⟩ / ψ(x)
    
    Parameters
    ----------
    samples : array, shape (n_samples, n_sites)
        Monte Carlo samples
    log_psi_fn : callable
        Function computing log ψ(x)
    hamiltonian : array
        Hamiltonian matrix
    
    Returns
    -------
    energy : float
        Energy estimate
    variance : float
        Variance of local energies
    """
    def local_energy(x):
        """Compute local energy for single sample."""
        log_psi_x = log_psi_fn(x)
        psi_x = jnp.exp(log_psi_x)
        
        # ⟨x|H|ψ⟩
        state_idx = binary_to_index(x)
        H_psi = 0.0
        dim = hamiltonian.shape[0]
        
        for y_idx in range(dim):
            if hamiltonian[state_idx, y_idx] != 0:
                y = index_to_binary(y_idx, len(x))
                log_psi_y = log_psi_fn(y)
                H_psi += hamiltonian[state_idx, y_idx] * jnp.exp(log_psi_y)
        
        return H_psi / psi_x
    
    local_energies = vmap(local_energy)(samples)
    
    energy = jnp.mean(jnp.real(local_energies))
    variance = jnp.var(jnp.real(local_energies))
    
    return energy, variance


def compute_correlation_function(
    samples: jnp.ndarray,
    operator_i: jnp.ndarray,
    operator_j: jnp.ndarray,
    site_i: int,
    site_j: int,
) -> float:
    """
    Compute two-point correlation function.
    
    C_{ij} = ⟨O_i O_j⟩ - ⟨O_i⟩⟨O_j⟩
    
    Parameters
    ----------
    samples : array
        Configuration samples
    operator_i, operator_j : array
        Local operators
    site_i, site_j : int
        Site indices
    
    Returns
    -------
    correlation : float
        Connected correlation function
    """
    def measure_operator(x, op, site):
        """Measure local operator on sample."""
        local_val = x[site]  # For diagonal operators
        return local_val * op[int(local_val), int(local_val)]
    
    O_i = vmap(lambda x: measure_operator(x, operator_i, site_i))(samples)
    O_j = vmap(lambda x: measure_operator(x, operator_j, site_j))(samples)
    
    correlation = jnp.mean(O_i * O_j) - jnp.mean(O_i) * jnp.mean(O_j)
    return correlation


def compute_structure_factor(
    samples: jnp.ndarray,
    q: jnp.ndarray,
    positions: jnp.ndarray,
) -> float:
    """
    Compute static structure factor.
    
    S(q) = (1/N) ∑_{ij} e^{iq·(r_i - r_j)} ⟨n_i n_j⟩
    
    Parameters
    ----------
    samples : array, shape (n_samples, n_sites)
        Density configurations
    q : array, shape (d,)
        Momentum vector
    positions : array, shape (n_sites, d)
        Site positions
    
    Returns
    -------
    S_q : float
        Structure factor at q
    """
    n_sites = samples.shape[1]
    
    def single_sample_sq(x):
        """Structure factor for single sample."""
        # Fourier transform of density
        n_q = jnp.sum(x * jnp.exp(1j * positions @ q))
        return jnp.abs(n_q) ** 2 / n_sites
    
    S_q = jnp.mean(vmap(single_sample_sq)(samples))
    return jnp.real(S_q)


def compute_entanglement_entropy(
    state: jnp.ndarray,
    subsystem_sites: jnp.ndarray,
    n_sites: int,
) -> float:
    """
    Compute von Neumann entanglement entropy.
    
    S_A = -Tr(ρ_A log ρ_A)
    
    Parameters
    ----------
    state : array, shape (2^n,)
        State vector in computational basis
    subsystem_sites : array
        Sites in subsystem A
    n_sites : int
        Total number of sites
    
    Returns
    -------
    entropy : float
        Entanglement entropy
    """
    dim = 2 ** n_sites
    n_A = len(subsystem_sites)
    n_B = n_sites - n_A
    dim_A = 2 ** n_A
    dim_B = 2 ** n_B
    
    # Reshape state into bipartite form
    # This is simplified - full implementation needs proper site ordering
    psi_matrix = state.reshape(dim_A, dim_B)
    
    # Reduced density matrix
    rho_A = psi_matrix @ psi_matrix.conj().T
    
    # Eigenvalues
    eigenvalues = jnp.linalg.eigvalsh(rho_A)
    eigenvalues = jnp.maximum(eigenvalues, 1e-15)  # Numerical stability
    
    # von Neumann entropy
    entropy = -jnp.sum(eigenvalues * jnp.log(eigenvalues))
    
    return entropy


def compute_v_score(
    energy: float,
    variance: float,
    exact_energy: float,
    n_params: int,
) -> float:
    """
    Compute variational score (v-score) for benchmarking.
    
    Following Wu et al., Science 386, 296 (2024).
    
    Parameters
    ----------
    energy : float
        Variational energy
    variance : float
        Energy variance
    exact_energy : float
        Exact ground state energy
    n_params : int
        Number of variational parameters
    
    Returns
    -------
    v_score : float
        Variational benchmark score
    """
    relative_error = jnp.abs(energy - exact_energy) / jnp.abs(exact_energy)
    normalized_variance = variance / (exact_energy ** 2)
    
    # v-score combines accuracy and variance
    v_score = -jnp.log10(relative_error + normalized_variance + 1e-10)
    
    return v_score


# Utility functions

def binary_to_index(x: jnp.ndarray) -> int:
    """Convert binary configuration to integer index."""
    return int(jnp.sum(x * (2 ** jnp.arange(len(x)))))


def index_to_binary(idx: int, n_sites: int) -> jnp.ndarray:
    """Convert integer index to binary configuration."""
    return jnp.array([(idx >> i) & 1 for i in range(n_sites)])


def spin_spin_correlation(
    samples: jnp.ndarray,
    site_i: int,
    site_j: int,
) -> float:
    """
    Compute spin-spin correlation ⟨S^z_i S^z_j⟩.
    
    For spin-1/2: S^z = (n - 1/2)
    """
    sz_i = samples[:, site_i] - 0.5
    sz_j = samples[:, site_j] - 0.5
    return jnp.mean(sz_i * sz_j)


def density_density_correlation(
    samples: jnp.ndarray,
    site_i: int,
    site_j: int,
) -> float:
    """Compute density-density correlation ⟨n_i n_j⟩ - ⟨n_i⟩⟨n_j⟩."""
    n_i = samples[:, site_i]
    n_j = samples[:, site_j]
    return jnp.mean(n_i * n_j) - jnp.mean(n_i) * jnp.mean(n_j)
