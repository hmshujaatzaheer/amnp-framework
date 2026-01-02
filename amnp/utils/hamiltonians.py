"""
Hamiltonians for Quantum Many-Body Systems
==========================================

Implementation of common Hamiltonians:
- Fermi-Hubbard model (Hermitian and dissipative)
- Heisenberg model
- Kagome lattice
"""

import jax.numpy as jnp
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class FermiHubbardHamiltonian:
    """
    Fermi-Hubbard Hamiltonian.
    
    H = -t ∑_{<ij>σ} (c†_{iσ} c_{jσ} + h.c.) + U ∑_i n_{i↑} n_{i↓}
    
    Parameters
    ----------
    Lx, Ly : int
        Lattice dimensions
    t : float
        Hopping amplitude
    U : float
        On-site interaction
    mu : float
        Chemical potential
    boundary : str
        'periodic' or 'open'
    """
    Lx: int
    Ly: int
    t: float = 1.0
    U: float = 4.0
    mu: float = 0.0
    boundary: str = 'periodic'
    
    @property
    def n_sites(self) -> int:
        return self.Lx * self.Ly
    
    @property
    def dim(self) -> int:
        """Hilbert space dimension (spinless for simplicity)."""
        return 2 ** self.n_sites
    
    def to_matrix(self) -> jnp.ndarray:
        """Construct full Hamiltonian matrix."""
        n = self.n_sites
        dim = self.dim
        H = jnp.zeros((dim, dim), dtype=complex)
        
        # Get neighbors
        neighbors = self._get_neighbors()
        
        for state in range(dim):
            # Diagonal: interaction and chemical potential
            for i in range(n):
                ni = (state >> i) & 1
                H = H.at[state, state].add(-self.mu * ni)
            
            # Off-diagonal: hopping
            for i, j in neighbors:
                ni = (state >> i) & 1
                nj = (state >> j) & 1
                
                # Hopping i -> j (if i occupied, j empty)
                if ni == 1 and nj == 0:
                    # Count fermions between i and j for sign
                    sign = self._jordan_wigner_sign(state, i, j)
                    new_state = state ^ (1 << i) ^ (1 << j)
                    H = H.at[new_state, state].add(-self.t * sign)
                    H = H.at[state, new_state].add(-self.t * sign)
        
        return H
    
    def _get_neighbors(self) -> List[Tuple[int, int]]:
        """Get list of nearest-neighbor pairs."""
        neighbors = []
        for y in range(self.Ly):
            for x in range(self.Lx):
                i = y * self.Lx + x
                
                # Right neighbor
                if x < self.Lx - 1 or self.boundary == 'periodic':
                    j = y * self.Lx + (x + 1) % self.Lx
                    neighbors.append((i, j))
                
                # Up neighbor
                if y < self.Ly - 1 or self.boundary == 'periodic':
                    j = ((y + 1) % self.Ly) * self.Lx + x
                    neighbors.append((i, j))
        
        return neighbors
    
    def _jordan_wigner_sign(self, state: int, i: int, j: int) -> int:
        """Compute Jordan-Wigner sign for hopping."""
        if i > j:
            i, j = j, i
        
        # Count occupied sites between i and j
        mask = ((1 << j) - 1) ^ ((1 << (i + 1)) - 1)
        n_between = bin(state & mask).count('1')
        return (-1) ** n_between


@dataclass
class DissipativeFermiHubbard(FermiHubbardHamiltonian):
    """
    Dissipative Fermi-Hubbard model.
    
    H_NH = H_FH + iΓ
    
    where Γ = γ ∑_i n_i represents particle loss.
    
    Parameters
    ----------
    gamma : float
        Dissipation rate
    """
    gamma: float = 0.1
    
    def to_matrix(self) -> jnp.ndarray:
        """Construct non-Hermitian Hamiltonian."""
        H_herm = super().to_matrix()
        
        # Add dissipator
        n = self.n_sites
        dim = self.dim
        Gamma = jnp.zeros((dim, dim), dtype=complex)
        
        for state in range(dim):
            n_total = bin(state).count('1')
            Gamma = Gamma.at[state, state].set(self.gamma * n_total)
        
        return H_herm + 1j * Gamma
    
    @property
    def is_hermitian(self) -> bool:
        return False


@dataclass
class HeisenbergHamiltonian:
    """
    Heisenberg spin model.
    
    H = J ∑_{<ij>} S_i · S_j = J ∑_{<ij>} (S^x_i S^x_j + S^y_i S^y_j + S^z_i S^z_j)
    
    Parameters
    ----------
    n_sites : int
        Number of spins
    J : float
        Exchange coupling (J > 0 antiferromagnetic)
    Jz : float, optional
        Anisotropy in z-direction
    """
    n_sites: int
    J: float = 1.0
    Jz: Optional[float] = None
    
    def __post_init__(self):
        if self.Jz is None:
            self.Jz = self.J
    
    @property
    def dim(self) -> int:
        return 2 ** self.n_sites
    
    def to_matrix(self) -> jnp.ndarray:
        """Construct Heisenberg Hamiltonian matrix."""
        n = self.n_sites
        dim = self.dim
        H = jnp.zeros((dim, dim), dtype=complex)
        
        # Pauli matrices
        sx = jnp.array([[0, 1], [1, 0]]) / 2
        sy = jnp.array([[0, -1j], [1j, 0]]) / 2
        sz = jnp.array([[1, 0], [0, -1]]) / 2
        
        for i in range(n - 1):
            j = i + 1
            
            # S^x_i S^x_j + S^y_i S^y_j (flip-flop)
            for state in range(dim):
                si = (state >> i) & 1
                sj = (state >> j) & 1
                
                # Z-Z interaction
                sz_i = 0.5 if si == 0 else -0.5
                sz_j = 0.5 if sj == 0 else -0.5
                H = H.at[state, state].add(self.Jz * sz_i * sz_j)
                
                # Flip-flop (S+S- + S-S+) / 2
                if si != sj:
                    new_state = state ^ (1 << i) ^ (1 << j)
                    H = H.at[new_state, state].add(0.5 * self.J)
        
        return H


@dataclass
class KagomeLattice:
    """
    Kagome lattice Hamiltonian.
    
    Supports both spin and fermionic models on kagome geometry.
    
    Parameters
    ----------
    Lx, Ly : int
        Unit cell dimensions
    model : str
        'heisenberg' or 'hubbard'
    """
    Lx: int
    Ly: int
    model: str = 'heisenberg'
    J: float = 1.0
    
    @property
    def n_sites(self) -> int:
        return 3 * self.Lx * self.Ly  # 3 sites per unit cell
    
    def get_kagome_neighbors(self) -> List[Tuple[int, int]]:
        """Get kagome lattice neighbor pairs."""
        neighbors = []
        
        for uy in range(self.Ly):
            for ux in range(self.Lx):
                # Sites in unit cell: A=0, B=1, C=2
                base = 3 * (uy * self.Lx + ux)
                A, B, C = base, base + 1, base + 2
                
                # Intra-cell bonds
                neighbors.extend([(A, B), (B, C), (C, A)])
                
                # Inter-cell bonds
                # A connects to C of left cell
                if ux > 0:
                    C_left = 3 * (uy * self.Lx + ux - 1) + 2
                    neighbors.append((A, C_left))
                
                # B connects to A of cell above
                if uy > 0:
                    A_below = 3 * ((uy - 1) * self.Lx + ux)
                    neighbors.append((B, A_below))
        
        return neighbors
    
    def to_matrix(self) -> jnp.ndarray:
        """Construct Hamiltonian matrix."""
        if self.model == 'heisenberg':
            return self._heisenberg_matrix()
        else:
            raise NotImplementedError(f"Model {self.model} not implemented")
    
    def _heisenberg_matrix(self) -> jnp.ndarray:
        """Heisenberg model on kagome."""
        n = self.n_sites
        dim = 2 ** n
        H = jnp.zeros((dim, dim), dtype=complex)
        
        neighbors = self.get_kagome_neighbors()
        
        for i, j in neighbors:
            for state in range(dim):
                si = (state >> i) & 1
                sj = (state >> j) & 1
                
                # Z-Z
                sz_i = 0.5 if si == 0 else -0.5
                sz_j = 0.5 if sj == 0 else -0.5
                H = H.at[state, state].add(self.J * sz_i * sz_j)
                
                # Flip-flop
                if si != sj:
                    new_state = state ^ (1 << i) ^ (1 << j)
                    H = H.at[new_state, state].add(0.5 * self.J)
        
        return H


def create_hamiltonian(
    name: str,
    **kwargs,
) -> jnp.ndarray:
    """
    Factory function to create Hamiltonians.
    
    Parameters
    ----------
    name : str
        Hamiltonian name: 'fermi_hubbard', 'dissipative_hubbard', 
        'heisenberg', 'kagome'
    **kwargs
        Parameters for the Hamiltonian
    
    Returns
    -------
    H : array
        Hamiltonian matrix
    """
    if name == 'fermi_hubbard':
        return FermiHubbardHamiltonian(**kwargs).to_matrix()
    elif name == 'dissipative_hubbard':
        return DissipativeFermiHubbard(**kwargs).to_matrix()
    elif name == 'heisenberg':
        return HeisenbergHamiltonian(**kwargs).to_matrix()
    elif name == 'kagome':
        return KagomeLattice(**kwargs).to_matrix()
    else:
        raise ValueError(f"Unknown Hamiltonian: {name}")
