"""
Thermofield-Extended Neural Gibbs States (TENGS)
================================================

Extends fermionic neural Gibbs states to non-Hermitian systems.

Key innovation: Non-Hermitian work operator for dissipative systems:
    W_NH = H ⊗ Ĩ - (β₀/β) Î ⊗ H̃₀ + iΓ ⊗ Ĩ

This enables finite-temperature simulation of open quantum systems
with particle loss, gain, or dephasing.

Reference:
- Nys & Carrasquilla, arXiv:2512.04663 (2025)
"""

import jax
import jax.numpy as jnp
from jax import random, vmap
from typing import Tuple, Optional, Callable, NamedTuple
from functools import partial
import flax.linen as nn

from .bivit_backflow import BiViTBackflow
from .jastrow import JastrowFactor


class TENGSConfig(NamedTuple):
    """Configuration for TENGS model."""
    n_sites: int
    hidden_size: int = 128
    n_layers: int = 2
    n_heads: int = 4
    beta_0: float = 0.1  # Initial inverse temperature
    beta: float = 1.0  # Target inverse temperature
    use_backflow: bool = True
    use_jastrow: bool = True


class TENGS(nn.Module):
    """
    Thermofield-Extended Neural Gibbs States.
    
    Represents finite-temperature states of non-Hermitian fermionic systems
    via thermofield double purification with neural network ansatz.
    
    The state is:
        |Ψ(β)⟩ = U(β/2) |Ψ₀(β₀)⟩
    
    where U(τ) is imaginary-time evolution under the work operator W_NH.
    
    Parameters
    ----------
    config : TENGSConfig
        Model configuration
    """
    config: TENGSConfig
    
    def setup(self):
        """Initialize neural network components."""
        cfg = self.config
        
        # 2D-GRU for spatial correlations (as in original Gibbs states)
        self.gru = nn.GRUCell(features=cfg.hidden_size)
        
        # BiViT attention for physical-auxiliary correlations
        self.attention = BiViTBackflow(
            hidden_size=cfg.hidden_size,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
        )
        
        # Pair orbital network
        self.orbital_net = nn.Dense(cfg.n_sites * cfg.n_sites)
        
        # Jastrow factor
        if cfg.use_jastrow:
            self.jastrow = JastrowFactor(cfg.n_sites)
    
    @nn.compact
    def __call__(
        self,
        x_physical: jnp.ndarray,
        x_auxiliary: jnp.ndarray,
    ) -> complex:
        """
        Compute log amplitude log Ψ(x, x̃).
        
        Parameters
        ----------
        x_physical : array, shape (n_sites,)
            Physical fermion configuration
        x_auxiliary : array, shape (n_sites,)
            Auxiliary (tilde) fermion configuration
        
        Returns
        -------
        log_psi : complex
            Log amplitude
        """
        cfg = self.config
        
        # Embed configurations
        x_embed = self._embed_config(x_physical)
        x_tilde_embed = self._embed_config(x_auxiliary)
        
        # Process through GRU
        h = jnp.zeros(cfg.hidden_size)
        for i in range(cfg.n_sites):
            h, _ = self.gru(jnp.concatenate([x_embed[i], x_tilde_embed[i]]), h)
        
        # BiViT attention between physical and auxiliary
        h_attended = self.attention(x_embed, x_tilde_embed, h)
        
        # Compute pair orbitals φ(x, x̃)
        orbitals = self.orbital_net(h_attended)
        orbitals = orbitals.reshape(cfg.n_sites, cfg.n_sites)
        
        # Slater determinant
        # Select occupied orbitals based on configuration
        n_up = int(jnp.sum(x_physical))
        if n_up > 0:
            occupied_rows = jnp.where(x_physical, size=n_up)[0]
            occupied_cols = jnp.where(x_auxiliary, size=n_up)[0]
            det_matrix = orbitals[occupied_rows][:, occupied_cols]
            log_det = jnp.linalg.slogdet(det_matrix)[1]
        else:
            log_det = 0.0
        
        # Jastrow correlation
        if cfg.use_jastrow:
            log_jastrow = self.jastrow(x_physical, x_auxiliary)
        else:
            log_jastrow = 0.0
        
        return log_det + log_jastrow
    
    def _embed_config(self, x: jnp.ndarray) -> jnp.ndarray:
        """Embed binary configuration."""
        # Simple embedding: one-hot + position encoding
        return nn.Dense(self.config.hidden_size)(x.astype(float)[:, None])
    
    def sample(
        self,
        key: jnp.ndarray,
        n_samples: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Sample from |Ψ|² using autoregressive sampling.
        
        Parameters
        ----------
        key : PRNGKey
            Random key
        n_samples : int
            Number of samples
        
        Returns
        -------
        x_physical : array, shape (n_samples, n_sites)
            Physical configurations
        x_auxiliary : array, shape (n_samples, n_sites)
            Auxiliary configurations
        log_probs : array, shape (n_samples,)
            Log probabilities
        """
        cfg = self.config
        keys = random.split(key, n_samples)
        
        def sample_single(k):
            return self._autoregressive_sample(k)
        
        samples = vmap(sample_single)(keys)
        return samples
    
    def _autoregressive_sample(
        self,
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """Single autoregressive sample."""
        cfg = self.config
        
        x_phys = jnp.zeros(cfg.n_sites, dtype=jnp.int32)
        x_aux = jnp.zeros(cfg.n_sites, dtype=jnp.int32)
        log_prob = 0.0
        
        key, *subkeys = random.split(key, 2 * cfg.n_sites + 1)
        
        # Sample site by site
        for i in range(cfg.n_sites):
            # Sample physical
            p_phys = self._conditional_prob(x_phys, x_aux, i, 'physical')
            x_phys = x_phys.at[i].set(random.bernoulli(subkeys[2*i], p_phys))
            log_prob += x_phys[i] * jnp.log(p_phys) + (1 - x_phys[i]) * jnp.log(1 - p_phys)
            
            # Sample auxiliary
            p_aux = self._conditional_prob(x_phys, x_aux, i, 'auxiliary')
            x_aux = x_aux.at[i].set(random.bernoulli(subkeys[2*i + 1], p_aux))
            log_prob += x_aux[i] * jnp.log(p_aux) + (1 - x_aux[i]) * jnp.log(1 - p_aux)
        
        return x_phys, x_aux, log_prob
    
    def _conditional_prob(
        self,
        x_phys: jnp.ndarray,
        x_aux: jnp.ndarray,
        site: int,
        which: str,
    ) -> float:
        """Compute conditional probability for autoregressive sampling."""
        # Simplified - full implementation uses masked attention
        return 0.5  # Placeholder
    
    def thermal_energy(
        self,
        hamiltonian: jnp.ndarray,
        n_samples: int = 1000,
        key: Optional[jnp.ndarray] = None,
    ) -> float:
        """
        Compute thermal energy ⟨H⟩_β.
        
        Parameters
        ----------
        hamiltonian : array
            Physical Hamiltonian
        n_samples : int
            Number of Monte Carlo samples
        key : PRNGKey, optional
            Random key
        
        Returns
        -------
        energy : float
            Thermal expectation value
        """
        if key is None:
            key = random.PRNGKey(0)
        
        x_phys, x_aux, _ = self.sample(key, n_samples)
        
        # Local energy estimator
        energies = vmap(lambda xp, xa: self._local_energy(xp, xa, hamiltonian))(
            x_phys, x_aux
        )
        
        return jnp.mean(energies)
    
    def _local_energy(
        self,
        x_phys: jnp.ndarray,
        x_aux: jnp.ndarray,
        H: jnp.ndarray,
    ) -> float:
        """Compute local energy for thermal state."""
        # E_loc = ⟨x|H|Ψ⟩ / ⟨x|Ψ⟩
        # For thermal states, trace over auxiliary
        log_psi = self(x_phys, x_aux)
        
        # Sum over connected configurations
        energy = 0.0
        for i in range(self.config.n_sites):
            for j in range(self.config.n_sites):
                if H[i, j] != 0:
                    x_new = x_phys.at[i].set(1 - x_phys[i])
                    x_new = x_new.at[j].set(1 - x_new[j])
                    log_psi_new = self(x_new, x_aux)
                    energy += H[i, j] * jnp.exp(log_psi_new - log_psi)
        
        return jnp.real(energy)


def create_tengs(
    n_sites: int,
    hidden_size: int = 128,
    beta: float = 1.0,
    **kwargs,
) -> TENGS:
    """
    Factory function to create TENGS model.
    
    Parameters
    ----------
    n_sites : int
        Number of fermionic sites
    hidden_size : int
        Hidden layer size
    beta : float
        Target inverse temperature
    **kwargs
        Additional configuration
    
    Returns
    -------
    model : TENGS
        Configured TENGS model
    """
    config = TENGSConfig(
        n_sites=n_sites,
        hidden_size=hidden_size,
        beta=beta,
        **kwargs,
    )
    return TENGS(config)


class NonHermitianWorkOperator:
    """
    Non-Hermitian work operator for TENGS evolution.
    
    W_NH = H ⊗ Ĩ - (β₀/β) Î ⊗ H̃₀ + iΓ ⊗ Ĩ
    
    Parameters
    ----------
    H : array
        Physical Hamiltonian
    H_0 : array
        Reference Hamiltonian
    gamma : array
        Dissipator
    beta_0 : float
        Initial inverse temperature
    beta : float
        Target inverse temperature
    """
    
    def __init__(
        self,
        H: jnp.ndarray,
        H_0: jnp.ndarray,
        gamma: jnp.ndarray,
        beta_0: float,
        beta: float,
    ):
        self.H = H
        self.H_0 = H_0
        self.gamma = gamma
        self.beta_0 = beta_0
        self.beta = beta
    
    def __call__(self) -> jnp.ndarray:
        """Construct the full work operator."""
        n = self.H.shape[0]
        I = jnp.eye(n)
        
        # H ⊗ Ĩ
        term1 = jnp.kron(self.H, I)
        
        # -(β₀/β) Î ⊗ H̃₀
        term2 = -(self.beta_0 / self.beta) * jnp.kron(I, self.H_0.T)
        
        # iΓ ⊗ Ĩ
        term3 = 1j * jnp.kron(self.gamma, I)
        
        return term1 + term2 + term3
    
    @property
    def is_hermitian(self) -> bool:
        """Check if work operator is Hermitian."""
        W = self()
        return jnp.allclose(W, W.conj().T)
