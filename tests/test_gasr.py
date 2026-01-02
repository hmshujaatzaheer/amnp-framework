"""
Tests for GASR Optimizer
========================

Unit tests for Geometry-Aware Stochastic Reconfiguration.
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random

from amnp.optimizers import GASR, compute_qgt, estimate_snr
from amnp.optimizers.snr_estimator import compute_effective_sample_size


class TestGASR:
    """Test suite for GASR optimizer."""
    
    def test_initialization(self):
        """Test GASR initialization."""
        optimizer = GASR(
            learning_rate=1e-3,
            snr_threshold=10.0,
            regularization=1e-4
        )
        
        assert optimizer.lr == 1e-3
        assert optimizer.tau == 10.0
        assert optimizer.lambda_reg == 1e-4
    
    def test_alpha_computation(self):
        """Test adaptive alpha parameter."""
        optimizer = GASR(snr_threshold=10.0)
        
        # Low SNR -> high alpha (more first-order)
        alpha_low = optimizer._compute_alpha(1.0)
        
        # High SNR -> low alpha (more curvature)
        alpha_high = optimizer._compute_alpha(100.0)
        
        assert alpha_low > alpha_high
        assert 0 <= alpha_low <= 1
        assert 0 <= alpha_high <= 1
    
    def test_state_initialization(self):
        """Test optimizer state initialization."""
        optimizer = GASR()
        params = jnp.zeros(10)
        state = optimizer.init(params)
        
        assert state.step == 0
        assert state.alpha == 0.5
        assert state.m.shape == (10,)
        assert state.v.shape == (10,)


class TestQGT:
    """Tests for Quantum Geometric Tensor computation."""
    
    def test_qgt_shape(self):
        """Test QGT output shape."""
        key = random.PRNGKey(0)
        n_params = 5
        n_samples = 100
        n_sites = 4
        
        params = random.normal(key, (n_params,))
        samples = random.bernoulli(key, 0.5, (n_samples, n_sites))
        
        def dummy_log_psi(p, x):
            return jnp.sum(p) * jnp.sum(x)
        
        S = compute_qgt(params, samples, dummy_log_psi)
        
        assert S.shape == (n_params, n_params)
    
    def test_qgt_symmetry(self):
        """Test QGT is Hermitian."""
        key = random.PRNGKey(42)
        n_params = 5
        n_samples = 100
        n_sites = 4
        
        params = random.normal(key, (n_params,))
        samples = random.bernoulli(key, 0.5, (n_samples, n_sites))
        
        def dummy_log_psi(p, x):
            return jnp.sum(p * jnp.arange(len(p))) * jnp.mean(x)
        
        S = compute_qgt(params, samples, dummy_log_psi)
        
        # Should be Hermitian (symmetric for real case)
        assert jnp.allclose(S, S.T, atol=1e-6)


class TestSNR:
    """Tests for SNR estimation."""
    
    def test_snr_positive(self):
        """Test SNR is always positive."""
        key = random.PRNGKey(0)
        gradients = random.normal(key, (100,))
        samples = random.bernoulli(key, 0.5, (1000, 10))
        
        snr = estimate_snr(gradients, samples)
        
        assert snr > 0
    
    def test_effective_sample_size(self):
        """Test effective sample size computation."""
        key = random.PRNGKey(0)
        
        # Independent samples should have N_eff ≈ N
        samples = random.normal(key, (1000, 5))
        n_eff = compute_effective_sample_size(samples)
        
        assert n_eff > 0
        assert n_eff <= 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
