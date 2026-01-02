"""
Visualization Utilities for AMNP Framework
==========================================

Plotting functions for:
- Training curves
- Energy landscapes
- Eigenvalue spectra
- Correlation functions
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import numpy as np


def plot_training_curve(
    energies: List[float],
    exact_energy: Optional[float] = None,
    title: str = "Training Progress",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot energy convergence during training.
    
    Parameters
    ----------
    energies : list
        Energy values at each step
    exact_energy : float, optional
        Exact ground state energy for reference
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(energies, 'b-', linewidth=2, label='Variational')
    
    if exact_energy is not None:
        ax.axhline(exact_energy, color='r', linestyle='--', 
                   label=f'Exact = {exact_energy:.4f}')
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_spectrum(
    eigenvalues: jnp.ndarray,
    gamma_max: Optional[float] = None,
    title: str = "Eigenvalue Spectrum",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot complex eigenvalue spectrum.
    
    Parameters
    ----------
    eigenvalues : array
        Complex eigenvalues
    gamma_max : float, optional
        NHTCT decay bound
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    real_parts = jnp.real(eigenvalues)
    imag_parts = jnp.imag(eigenvalues)
    
    # Color by decay rate
    colors = jnp.abs(imag_parts)
    scatter = ax.scatter(real_parts, imag_parts, c=colors, 
                         cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter, ax=ax, label='|Im(λ)|')
    
    if gamma_max is not None:
        ax.axhline(gamma_max, color='red', linestyle='--', 
                   label=f'Γ_max = {gamma_max:.2f}')
        ax.axhline(-gamma_max, color='red', linestyle='--')
        ax.fill_between(ax.get_xlim(), -gamma_max, gamma_max, 
                        alpha=0.1, color='green', label='Kept region')
    
    ax.set_xlabel('Re(λ)', fontsize=12)
    ax.set_ylabel('Im(λ)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_gasr_alpha(
    alphas: List[float],
    snrs: Optional[List[float]] = None,
    title: str = "GASR Adaptive Parameter",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot GASR alpha parameter evolution.
    
    Parameters
    ----------
    alphas : list
        Alpha values during training
    snrs : list, optional
        SNR values during training
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib Figure
    """
    if snrs is not None:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 4))
        axes = [axes]
    
    # Alpha plot
    axes[0].plot(alphas, 'g-', linewidth=2)
    axes[0].set_ylabel('α (interpolation)', fontsize=12)
    axes[0].set_ylim(0, 1)
    axes[0].axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    axes[0].fill_between(range(len(alphas)), 0, alphas, alpha=0.3, color='green')
    axes[0].set_title(title, fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Annotations
    axes[0].annotate('More SR', xy=(0.02, 0.1), xycoords='axes fraction', fontsize=10)
    axes[0].annotate('More Adam', xy=(0.02, 0.9), xycoords='axes fraction', fontsize=10)
    
    if snrs is not None:
        axes[1].semilogy(snrs, 'b-', linewidth=2)
        axes[1].set_ylabel('SNR (log scale)', fontsize=12)
        axes[1].set_xlabel('Training Step', fontsize=12)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[0].set_xlabel('Training Step', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_correlation_function(
    distances: jnp.ndarray,
    correlations: jnp.ndarray,
    correlation_type: str = "spin-spin",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot correlation function vs distance.
    
    Parameters
    ----------
    distances : array
        Site distances
    correlations : array
        Correlation values
    correlation_type : str
        Type of correlation
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(distances, correlations, 'bo-', markersize=8, linewidth=2)
    
    # Fit exponential decay
    if len(distances) > 2:
        try:
            from scipy.optimize import curve_fit
            
            def exp_decay(r, A, xi):
                return A * np.exp(-r / xi)
            
            popt, _ = curve_fit(exp_decay, np.array(distances), 
                               np.array(correlations), p0=[1, 1])
            
            r_fit = np.linspace(min(distances), max(distances), 100)
            ax.plot(r_fit, exp_decay(r_fit, *popt), 'r--', 
                   label=f'ξ = {popt[1]:.2f}')
            ax.legend()
        except:
            pass
    
    ax.set_xlabel('Distance r', fontsize=12)
    ax.set_ylabel(f'⟨{correlation_type}⟩', fontsize=12)
    
    if title is None:
        title = f'{correlation_type.capitalize()} Correlation Function'
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_density_matrix(
    rho: jnp.ndarray,
    title: str = "Density Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot density matrix as heatmap.
    
    Parameters
    ----------
    rho : array
        Density matrix
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Real part
    im0 = axes[0].imshow(jnp.real(rho), cmap='RdBu', aspect='auto')
    axes[0].set_title('Re(ρ)', fontsize=12)
    plt.colorbar(im0, ax=axes[0])
    
    # Imaginary part
    im1 = axes[1].imshow(jnp.imag(rho), cmap='RdBu', aspect='auto')
    axes[1].set_title('Im(ρ)', fontsize=12)
    plt.colorbar(im1, ax=axes[1])
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
