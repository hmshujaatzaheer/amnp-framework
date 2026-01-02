"""
Jastrow Correlation Factor
==========================

Jastrow factors capture explicit two-body correlations:
    J(x) = exp(-∑_{i<j} v_{ij} n_i n_j)

Combined with Slater determinants for accurate fermionic states.
"""

import jax
import jax.numpy as jnp
from typing import Optional
import flax.linen as nn


class JastrowFactor(nn.Module):
    """
    Neural Jastrow correlation factor.
    
    Parameterizes pairwise correlations using a neural network.
    
    Parameters
    ----------
    n_sites : int
        Number of sites
    use_neural : bool
        If True, use neural network; else use explicit v_ij
    hidden_size : int
        Hidden size for neural Jastrow
    """
    n_sites: int
    use_neural: bool = True
    hidden_size: int = 32
    
    @nn.compact
    def __call__(
        self,
        x_physical: jnp.ndarray,
        x_auxiliary: Optional[jnp.ndarray] = None,
    ) -> float:
        """
        Compute log Jastrow factor.
        
        Parameters
        ----------
        x_physical : array, shape (n_sites,)
            Physical configuration
        x_auxiliary : array, optional
            Auxiliary configuration (for TENGS)
        
        Returns
        -------
        log_jastrow : float
            Log of Jastrow factor
        """
        if self.use_neural:
            return self._neural_jastrow(x_physical, x_auxiliary)
        else:
            return self._explicit_jastrow(x_physical)
    
    def _neural_jastrow(
        self,
        x_physical: jnp.ndarray,
        x_auxiliary: Optional[jnp.ndarray],
    ) -> float:
        """Neural network Jastrow."""
        # Embed configuration
        x = x_physical.astype(float)
        if x_auxiliary is not None:
            x = jnp.concatenate([x, x_auxiliary.astype(float)])
        
        # MLP
        h = nn.Dense(self.hidden_size)(x)
        h = nn.tanh(h)
        h = nn.Dense(self.hidden_size)(h)
        h = nn.tanh(h)
        log_j = nn.Dense(1)(h)
        
        return log_j.squeeze()
    
    def _explicit_jastrow(self, x: jnp.ndarray) -> float:
        """Explicit pairwise Jastrow."""
        # Learnable pairwise parameters
        v = self.param(
            'v_ij',
            nn.initializers.normal(0.01),
            (self.n_sites, self.n_sites)
        )
        
        # Make symmetric
        v = 0.5 * (v + v.T)
        
        # Compute -∑_{i<j} v_{ij} n_i n_j
        n = x.astype(float)
        log_jastrow = -0.5 * n @ v @ n
        
        return log_jastrow


class ExtendedJastrow(nn.Module):
    """
    Extended Jastrow with three-body and backflow terms.
    
    J(x) = J_2(x) + J_3(x) + J_bf(x)
    """
    n_sites: int
    include_three_body: bool = True
    include_backflow: bool = True
    hidden_size: int = 64
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> float:
        """Compute extended Jastrow factor."""
        log_j = 0.0
        
        # Two-body
        log_j += JastrowFactor(
            self.n_sites, use_neural=True, hidden_size=self.hidden_size
        )(x)
        
        # Three-body (via neural network)
        if self.include_three_body:
            log_j += self._three_body(x)
        
        # Backflow correction
        if self.include_backflow:
            log_j += self._backflow_correction(x)
        
        return log_j
    
    def _three_body(self, x: jnp.ndarray) -> float:
        """Three-body correlations via neural network."""
        x_float = x.astype(float)
        
        # Create pair features
        n = self.n_sites
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                pairs.append(x_float[i] * x_float[j])
        
        pair_features = jnp.array(pairs)
        
        # MLP on pair features
        h = nn.Dense(self.hidden_size)(pair_features)
        h = nn.gelu(h)
        h = nn.Dense(self.hidden_size // 2)(h)
        h = nn.gelu(h)
        log_j3 = nn.Dense(1)(h)
        
        return log_j3.squeeze()
    
    def _backflow_correction(self, x: jnp.ndarray) -> float:
        """Backflow correction term."""
        x_float = x.astype(float)
        
        # Backflow coordinates
        backflow = nn.Dense(self.n_sites)(x_float)
        backflow = nn.tanh(backflow)
        
        # Correction from backflow
        correction = nn.Dense(self.hidden_size)(backflow)
        correction = nn.gelu(correction)
        log_bf = nn.Dense(1)(correction)
        
        return log_bf.squeeze()


class SpinJastrow(nn.Module):
    """
    Jastrow factor for spin systems.
    
    J(σ) = exp(-∑_{i<j} v_{ij} σ_i σ_j)
    """
    n_sites: int
    
    @nn.compact
    def __call__(self, spins: jnp.ndarray) -> float:
        """
        Compute spin Jastrow.
        
        Parameters
        ----------
        spins : array, shape (n_sites,)
            Spin configuration (±1)
        
        Returns
        -------
        log_jastrow : float
        """
        v = self.param(
            'v_ij',
            nn.initializers.normal(0.01),
            (self.n_sites, self.n_sites)
        )
        v = 0.5 * (v + v.T)
        
        s = spins.astype(float)
        log_jastrow = -0.5 * s @ v @ s
        
        return log_jastrow
