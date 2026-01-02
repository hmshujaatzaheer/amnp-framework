"""
Example: NHTCT for Non-Hermitian Dynamics
=========================================

Demonstrates Non-Hermitian Trotter-Consistent Truncation (NHTCT)
on a dissipative Fermi-Hubbard model with particle loss.

This example shows how NHTCT extends Majorana string propagation
to handle complex eigenvalue spectra via decay-bounded truncation.
"""

import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

from amnp.truncation import NHTCT, create_nhtct, compute_gamma_max, analyze_spectrum
from amnp.utils import DissipativeFermiHubbard


def main():
    print("=" * 60)
    print("NHTCT: Non-Hermitian Trotter-Consistent Truncation Demo")
    print("=" * 60)
    
    # System parameters
    Lx, Ly = 2, 2  # 2x2 lattice (4 sites)
    t = 1.0        # Hopping
    U = 4.0        # Interaction
    gamma = 0.1    # Dissipation rate
    
    print(f"\nSystem: {Lx}x{Ly} Dissipative Fermi-Hubbard")
    print(f"  Hopping t = {t}")
    print(f"  Interaction U = {U}")
    print(f"  Dissipation γ = {gamma}")
    
    # Create non-Hermitian Hamiltonian
    hamiltonian = DissipativeFermiHubbard(
        Lx=Lx, Ly=Ly, t=t, U=U, gamma=gamma
    )
    H = hamiltonian.to_matrix()
    
    print(f"\nHamiltonian dimension: {H.shape[0]}x{H.shape[1]}")
    print(f"Is Hermitian: {hamiltonian.is_hermitian}")
    
    # Analyze spectrum
    eigenvalues = jnp.linalg.eigvals(H)
    print(f"\nSpectrum analysis:")
    print(f"  Real part range: [{jnp.min(jnp.real(eigenvalues)):.3f}, "
          f"{jnp.max(jnp.real(eigenvalues)):.3f}]")
    print(f"  Imag part range: [{jnp.min(jnp.imag(eigenvalues)):.3f}, "
          f"{jnp.max(jnp.imag(eigenvalues)):.3f}]")
    
    # NHTCT parameters
    max_weight = 4
    trotter_error = 1e-4
    time_step = 0.01
    
    gamma_max = compute_gamma_max(time_step, trotter_error)
    print(f"\nNHTCT Parameters:")
    print(f"  Max weight S = {max_weight}")
    print(f"  Trotter error ε = {trotter_error}")
    print(f"  Time step δτ = {time_step}")
    print(f"  Max decay rate Γ_max = {gamma_max:.3f}")
    
    # Create NHTCT scheme
    nhtct = create_nhtct(
        max_weight=max_weight,
        trotter_error=trotter_error,
        time_step=time_step
    )
    
    # Analyze truncation
    spectrum_stats = analyze_spectrum(eigenvalues, gamma_max)
    print(f"\nSpectrum truncation analysis:")
    print(f"  Total eigenvalues: {spectrum_stats['n_total']}")
    print(f"  Stable (real): {spectrum_stats['n_stable']}")
    print(f"  Would be kept: {spectrum_stats['n_kept']}")
    print(f"  Would be truncated: {spectrum_stats['n_truncated']}")
    print(f"  Truncation fraction: {spectrum_stats['truncation_fraction']:.1%}")
    
    # Time evolution
    print("\n" + "-" * 60)
    print("Running time evolution...")
    
    # Initial state: half-filled Fock state
    n_sites = Lx * Ly
    dim = 2 ** n_sites
    initial_state = jnp.zeros(dim, dtype=complex)
    # |1010⟩ state (alternating occupation)
    initial_idx = sum(2**i for i in range(0, n_sites, 2))
    initial_state = initial_state.at[initial_idx].set(1.0)
    
    t_final = 1.0
    times, states = nhtct.evolve(
        H, initial_state, t_final,
        callback=lambda step, t, psi: None
    )
    
    print(f"Evolution completed: {len(times)} time steps")
    
    # Compute observables
    densities = []
    for psi in states:
        # Total density
        density = 0
        for state_idx in range(dim):
            n_particles = bin(state_idx).count('1')
            density += n_particles * jnp.abs(psi[state_idx])**2
        densities.append(float(jnp.real(density)))
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Eigenvalue spectrum
    ax1 = axes[0]
    ax1.scatter(jnp.real(eigenvalues), jnp.imag(eigenvalues), 
                c='blue', alpha=0.6, s=50)
    ax1.axhline(-gamma_max, color='red', linestyle='--', 
                label=f'Γ_max = {gamma_max:.2f}')
    ax1.axhline(gamma_max, color='red', linestyle='--')
    ax1.set_xlabel('Re(λ)')
    ax1.set_ylabel('Im(λ)')
    ax1.set_title('Non-Hermitian Spectrum with NHTCT Bounds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Density evolution
    ax2 = axes[1]
    ax2.plot(times, densities, 'b-', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Total Particle Number')
    ax2.set_title('Particle Number Under Dissipation')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nhtct_dynamics.png', dpi=150)
    print("\nPlot saved to 'nhtct_dynamics.png'")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Initial particles: {densities[0]:.2f}")
    print(f"  Final particles: {densities[-1]:.2f}")
    print(f"  Particle loss: {(1 - densities[-1]/densities[0])*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
