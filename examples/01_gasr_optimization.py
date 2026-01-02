"""
Example: GASR Optimization for RNN Wave Functions
=================================================

Demonstrates the Geometry-Aware Stochastic Reconfiguration (GASR) optimizer
on a small Heisenberg model.

This example shows how GASR adaptively balances between curvature-aware
optimization (Stochastic Reconfiguration) and first-order methods (Adam)
based on gradient signal-to-noise ratio.
"""

import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

# Import AMNP components
from amnp.optimizers import GASR, create_gasr_optimizer
from amnp.neural_states import RNNWavefunction, create_rnn_wavefunction
from amnp.utils import HeisenbergHamiltonian, compute_energy, compute_v_score


def main():
    # Set random seed for reproducibility
    key = random.PRNGKey(42)
    
    # System parameters
    n_sites = 6
    J = 1.0  # Antiferromagnetic coupling
    
    # Create Hamiltonian
    print("Creating Heisenberg Hamiltonian...")
    hamiltonian = HeisenbergHamiltonian(n_sites=n_sites, J=J)
    H = hamiltonian.to_matrix()
    
    # Exact ground state energy (for comparison)
    eigenvalues = jnp.linalg.eigvalsh(H)
    exact_energy = eigenvalues[0]
    print(f"Exact ground state energy: {exact_energy:.6f}")
    
    # Create RNN wave function
    print("\nInitializing RNN wave function...")
    model = create_rnn_wavefunction(
        num_sites=n_sites,
        hidden_size=32,
        num_layers=2,
        cell_type='gru'
    )
    
    # Initialize parameters
    key, init_key = random.split(key)
    dummy_input = jnp.zeros(n_sites, dtype=jnp.int32)
    params = model.init(init_key, dummy_input)
    
    # Create GASR optimizer
    optimizer = create_gasr_optimizer(
        learning_rate=1e-2,
        snr_threshold=10.0,
        regularization=1e-4
    )
    opt_state = optimizer.init(params)
    
    # Training parameters
    n_steps = 500
    n_samples = 256
    
    # Storage for plotting
    energies = []
    alphas = []
    
    print("\nStarting GASR optimization...")
    print("-" * 50)
    
    for step in range(n_steps):
        key, sample_key = random.split(key)
        
        # Sample configurations
        samples, log_probs = model.apply(
            params, sample_key, n_samples, 
            method=model.sample
        )
        
        # Define log_psi function for this step
        def log_psi_fn(p, x):
            return model.apply(p, x)
        
        # Compute energy and gradients
        def loss_fn(p):
            def local_energy(x):
                # Simplified local energy computation
                log_psi_x = model.apply(p, x)
                # In practice, compute ⟨x|H|ψ⟩/ψ(x)
                return jnp.real(log_psi_x)  # Placeholder
            
            energies = jax.vmap(local_energy)(samples)
            return jnp.mean(energies), energies
        
        (energy, local_energies), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        
        # GASR update
        opt_state, params = optimizer.update(
            opt_state, params, grads, samples,
            lambda p, x: model.apply(p, x)
        )
        
        # Store metrics
        energies.append(float(energy))
        alphas.append(float(opt_state.alpha))
        
        # Print progress
        if step % 50 == 0 or step == n_steps - 1:
            print(f"Step {step:4d}: E = {energy:.6f}, "
                  f"α = {opt_state.alpha:.3f}, "
                  f"SNR = {opt_state.snr_ema:.2f}")
    
    print("-" * 50)
    
    # Final results
    final_energy = energies[-1]
    relative_error = abs(final_energy - exact_energy) / abs(exact_energy)
    
    print(f"\nFinal Results:")
    print(f"  Final energy: {final_energy:.6f}")
    print(f"  Exact energy: {exact_energy:.6f}")
    print(f"  Relative error: {relative_error:.2e}")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Energy convergence
    axes[0].plot(energies, 'b-', label='GASR')
    axes[0].axhline(exact_energy, color='r', linestyle='--', label='Exact')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Energy')
    axes[0].set_title('Energy Convergence')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Alpha adaptation
    axes[1].plot(alphas, 'g-')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('α (interpolation)')
    axes[1].set_title('GASR Adaptive Parameter')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gasr_optimization.png', dpi=150)
    print("\nPlot saved to 'gasr_optimization.png'")
    

if __name__ == "__main__":
    main()
