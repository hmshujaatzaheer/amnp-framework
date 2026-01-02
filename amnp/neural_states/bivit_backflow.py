"""
BiViT Backflow Neural Network
=============================

Bidirectional Vision Transformer with backflow for neural quantum states.

Architecture:
- 2D-GRU for local correlations
- Multi-head attention for non-local correlations
- Backflow transformation for orbital modification

Reference:
- Nys & Carrasquilla, arXiv:2512.04663 (2025)
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Optional
import flax.linen as nn


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer."""
    hidden_size: int
    n_heads: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Apply multi-head attention.
        
        Parameters
        ----------
        query, key, value : array, shape (..., seq_len, hidden_size)
            Input tensors
        mask : array, optional
            Attention mask
        deterministic : bool
            If False, apply dropout
        
        Returns
        -------
        output : array
            Attended output
        """
        head_dim = self.hidden_size // self.n_heads
        
        # Project to Q, K, V
        Q = nn.Dense(self.hidden_size)(query)
        K = nn.Dense(self.hidden_size)(key)
        V = nn.Dense(self.hidden_size)(value)
        
        # Reshape for multi-head
        batch_size = query.shape[0] if query.ndim > 2 else 1
        seq_len = query.shape[-2]
        
        Q = Q.reshape(-1, seq_len, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(-1, seq_len, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(-1, seq_len, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scale = jnp.sqrt(head_dim)
        attn_weights = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / scale
        
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -1e9)
        
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        
        if not deterministic:
            attn_weights = nn.Dropout(self.dropout_rate)(
                attn_weights, deterministic=False
            )
        
        # Apply attention to values
        output = jnp.matmul(attn_weights, V)
        
        # Reshape back
        output = output.transpose(0, 2, 1, 3).reshape(-1, seq_len, self.hidden_size)
        
        # Final projection
        output = nn.Dense(self.hidden_size)(output)
        
        return output.squeeze(0) if query.ndim == 2 else output


class BiViTBlock(nn.Module):
    """Single BiViT transformer block."""
    hidden_size: int
    n_heads: int
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(
        self,
        x_physical: jnp.ndarray,
        x_auxiliary: jnp.ndarray,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Bidirectional attention between physical and auxiliary.
        
        Physical attends to auxiliary and vice versa.
        """
        # Self-attention within each
        x_phys_self = MultiHeadAttention(
            self.hidden_size, self.n_heads, self.dropout_rate
        )(x_physical, x_physical, x_physical, deterministic=deterministic)
        
        x_aux_self = MultiHeadAttention(
            self.hidden_size, self.n_heads, self.dropout_rate
        )(x_auxiliary, x_auxiliary, x_auxiliary, deterministic=deterministic)
        
        # Cross-attention
        x_phys_cross = MultiHeadAttention(
            self.hidden_size, self.n_heads, self.dropout_rate
        )(x_physical, x_auxiliary, x_auxiliary, deterministic=deterministic)
        
        x_aux_cross = MultiHeadAttention(
            self.hidden_size, self.n_heads, self.dropout_rate
        )(x_auxiliary, x_physical, x_physical, deterministic=deterministic)
        
        # Combine with residual
        x_physical = nn.LayerNorm()(x_physical + x_phys_self + x_phys_cross)
        x_auxiliary = nn.LayerNorm()(x_auxiliary + x_aux_self + x_aux_cross)
        
        # MLP
        mlp_hidden = int(self.hidden_size * self.mlp_ratio)
        
        x_phys_mlp = nn.Dense(mlp_hidden)(x_physical)
        x_phys_mlp = nn.gelu(x_phys_mlp)
        x_phys_mlp = nn.Dense(self.hidden_size)(x_phys_mlp)
        
        x_aux_mlp = nn.Dense(mlp_hidden)(x_auxiliary)
        x_aux_mlp = nn.gelu(x_aux_mlp)
        x_aux_mlp = nn.Dense(self.hidden_size)(x_aux_mlp)
        
        x_physical = nn.LayerNorm()(x_physical + x_phys_mlp)
        x_auxiliary = nn.LayerNorm()(x_auxiliary + x_aux_mlp)
        
        return x_physical, x_auxiliary


class BiViTBackflow(nn.Module):
    """
    BiViT with backflow transformation.
    
    Combines bidirectional attention with backflow orbital modification.
    
    Parameters
    ----------
    hidden_size : int
        Hidden dimension
    n_heads : int
        Number of attention heads
    n_layers : int
        Number of BiViT blocks
    """
    hidden_size: int
    n_heads: int
    n_layers: int = 2
    
    @nn.compact
    def __call__(
        self,
        x_physical: jnp.ndarray,
        x_auxiliary: jnp.ndarray,
        context: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Apply BiViT with backflow.
        
        Parameters
        ----------
        x_physical : array, shape (n_sites, hidden_size)
            Physical site embeddings
        x_auxiliary : array, shape (n_sites, hidden_size)
            Auxiliary site embeddings
        context : array, shape (hidden_size,)
            Global context from GRU
        
        Returns
        -------
        output : array, shape (hidden_size,)
            Backflow-modified representation
        """
        # Add context to embeddings
        x_physical = x_physical + context[None, :]
        x_auxiliary = x_auxiliary + context[None, :]
        
        # Apply BiViT blocks
        for _ in range(self.n_layers):
            x_physical, x_auxiliary = BiViTBlock(
                self.hidden_size, self.n_heads
            )(x_physical, x_auxiliary)
        
        # Pool and combine
        phys_pooled = jnp.mean(x_physical, axis=0)
        aux_pooled = jnp.mean(x_auxiliary, axis=0)
        
        combined = jnp.concatenate([phys_pooled, aux_pooled, context])
        
        # Backflow transformation
        backflow = nn.Dense(self.hidden_size)(combined)
        backflow = nn.tanh(backflow)
        backflow = nn.Dense(self.hidden_size)(backflow)
        
        return backflow


class GRU2D(nn.Module):
    """
    2D GRU for processing lattice configurations.
    
    Processes sites in a snake pattern for 2D systems.
    """
    hidden_size: int
    
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        Lx: int,
        Ly: int,
    ) -> jnp.ndarray:
        """
        Process 2D configuration through GRU.
        
        Parameters
        ----------
        x : array, shape (Lx * Ly,)
            Flattened configuration
        Lx, Ly : int
            Lattice dimensions
        
        Returns
        -------
        h : array, shape (hidden_size,)
            Final hidden state
        """
        gru_cell = nn.GRUCell(features=self.hidden_size)
        embed = nn.Dense(self.hidden_size)
        
        h = jnp.zeros(self.hidden_size)
        
        # Snake pattern through lattice
        for row in range(Ly):
            if row % 2 == 0:
                cols = range(Lx)
            else:
                cols = range(Lx - 1, -1, -1)
            
            for col in cols:
                idx = row * Lx + col
                x_embed = embed(x[idx:idx+1])
                h, _ = gru_cell(x_embed, h)
        
        return h
