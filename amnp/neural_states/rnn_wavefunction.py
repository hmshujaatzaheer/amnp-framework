"""
Recurrent Neural Network Wave Function
======================================

Autoregressive neural quantum state using RNN/GRU architecture.

The wave function is factorized as:
    ψ(x) = ∏_i p(x_i | x_{<i})

where each conditional is parameterized by an RNN.

Reference:
- Hibat-Allah et al., Commun. Phys. 8, 308 (2025)
"""

import jax
import jax.numpy as jnp
from jax import random, vmap
from typing import Tuple, Optional
import flax.linen as nn


class RNNWavefunction(nn.Module):
    """
    RNN-based neural quantum state.
    
    Parameters
    ----------
    hidden_size : int
        RNN hidden dimension
    num_layers : int
        Number of stacked RNN layers
    num_sites : int
        Number of lattice sites
    cell_type : str
        'gru' or 'lstm'
    """
    hidden_size: int
    num_layers: int
    num_sites: int
    cell_type: str = 'gru'
    
    def setup(self):
        """Initialize RNN cells."""
        if self.cell_type == 'gru':
            self.cells = [nn.GRUCell(features=self.hidden_size) 
                         for _ in range(self.num_layers)]
        elif self.cell_type == 'lstm':
            self.cells = [nn.LSTMCell(features=self.hidden_size)
                         for _ in range(self.num_layers)]
        else:
            raise ValueError(f"Unknown cell type: {self.cell_type}")
        
        # Input embedding
        self.embed = nn.Dense(self.hidden_size)
        
        # Output head for conditional probabilities
        self.output_head = nn.Dense(2)  # Binary output
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> complex:
        """
        Compute log amplitude log ψ(x).
        
        Parameters
        ----------
        x : array, shape (num_sites,)
            Binary configuration
        
        Returns
        -------
        log_psi : complex
            Log amplitude
        """
        log_amp = 0.0
        log_phase = 0.0
        
        # Initialize hidden states
        h = [jnp.zeros(self.hidden_size) for _ in range(self.num_layers)]
        
        # Process sequentially
        for i in range(self.num_sites):
            # Get input (shifted: use x_{i-1} to predict x_i)
            if i == 0:
                x_in = jnp.zeros(1)
            else:
                x_in = x[i-1:i].astype(float)
            
            # Embed input
            x_embed = self.embed(x_in)
            
            # Pass through RNN layers
            for layer, cell in enumerate(self.cells):
                h[layer], _ = cell(x_embed if layer == 0 else h[layer-1], h[layer])
            
            # Compute conditional log probability
            logits = self.output_head(h[-1])
            log_probs = jax.nn.log_softmax(logits)
            
            # Accumulate log amplitude
            log_amp += log_probs[int(x[i])]
        
        return log_amp + 1j * log_phase
    
    def log_prob(self, x: jnp.ndarray) -> float:
        """Compute log probability log |ψ(x)|²."""
        log_psi = self(x)
        return 2 * jnp.real(log_psi)
    
    def sample(
        self,
        key: jnp.ndarray,
        num_samples: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate samples from |ψ|² using autoregressive sampling.
        
        Parameters
        ----------
        key : PRNGKey
            Random key
        num_samples : int
            Number of samples to generate
        
        Returns
        -------
        samples : array, shape (num_samples, num_sites)
            Sampled configurations
        log_probs : array, shape (num_samples,)
            Log probabilities of samples
        """
        keys = random.split(key, num_samples)
        
        def sample_single(k):
            return self._sample_single(k)
        
        results = vmap(sample_single)(keys)
        return results
    
    def _sample_single(self, key: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
        """Sample single configuration."""
        x = jnp.zeros(self.num_sites, dtype=jnp.int32)
        log_prob = 0.0
        
        h = [jnp.zeros(self.hidden_size) for _ in range(self.num_layers)]
        keys = random.split(key, self.num_sites)
        
        for i in range(self.num_sites):
            # Get input
            if i == 0:
                x_in = jnp.zeros(1)
            else:
                x_in = x[i-1:i].astype(float)
            
            # Forward pass
            x_embed = self.embed(x_in)
            for layer, cell in enumerate(self.cells):
                h[layer], _ = cell(x_embed if layer == 0 else h[layer-1], h[layer])
            
            # Sample from conditional
            logits = self.output_head(h[-1])
            probs = jax.nn.softmax(logits)
            
            sampled = random.categorical(keys[i], logits)
            x = x.at[i].set(sampled)
            log_prob += jnp.log(probs[sampled])
        
        return x, log_prob
    
    def conditional_probs(
        self,
        x_partial: jnp.ndarray,
        site: int,
    ) -> jnp.ndarray:
        """
        Compute conditional probability distribution at site.
        
        Parameters
        ----------
        x_partial : array
            Partial configuration up to site
        site : int
            Site index
        
        Returns
        -------
        probs : array, shape (2,)
            Probability distribution [p(0), p(1)]
        """
        h = [jnp.zeros(self.hidden_size) for _ in range(self.num_layers)]
        
        # Process up to site
        for i in range(site):
            if i == 0:
                x_in = jnp.zeros(1)
            else:
                x_in = x_partial[i-1:i].astype(float)
            
            x_embed = self.embed(x_in)
            for layer, cell in enumerate(self.cells):
                h[layer], _ = cell(x_embed if layer == 0 else h[layer-1], h[layer])
        
        # Get final input
        if site == 0:
            x_in = jnp.zeros(1)
        else:
            x_in = x_partial[site-1:site].astype(float)
        
        x_embed = self.embed(x_in)
        for layer, cell in enumerate(self.cells):
            h[layer], _ = cell(x_embed if layer == 0 else h[layer-1], h[layer])
        
        logits = self.output_head(h[-1])
        return jax.nn.softmax(logits)


def create_rnn_wavefunction(
    num_sites: int,
    hidden_size: int = 64,
    num_layers: int = 2,
    cell_type: str = 'gru',
) -> RNNWavefunction:
    """
    Factory function to create RNN wavefunction.
    
    Parameters
    ----------
    num_sites : int
        Number of lattice sites
    hidden_size : int
        Hidden dimension
    num_layers : int
        Number of RNN layers
    cell_type : str
        'gru' or 'lstm'
    
    Returns
    -------
    model : RNNWavefunction
        Configured model
    """
    return RNNWavefunction(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_sites=num_sites,
        cell_type=cell_type,
    )
