"""
Majorana String Algebra
=======================

Implementation of Majorana string representation for fermionic systems.

A Majorana string μ(v) = γ_1^{v_1} γ_2^{v_2} ... γ_{2N}^{v_{2N}}
where v ∈ {0,1}^{2N} indicates which Majorana operators appear.

The string weight w_s(v) counts unpaired Majoranas.

Reference:
- D'Anna, Nys, Carrasquilla, arXiv:2511.02809 (2025)
"""

import jax
import jax.numpy as jnp
from typing import Tuple, List, NamedTuple, Optional
from functools import partial


class MajoranaString(NamedTuple):
    """
    Representation of a Majorana string.
    
    Attributes
    ----------
    indices : array of int
        Binary string v indicating which Majoranas appear
    coefficient : complex
        Complex coefficient
    n_sites : int
        Number of fermionic sites (N, so 2N Majoranas)
    """
    indices: jnp.ndarray  # Binary vector of length 2N
    coefficient: complex
    n_sites: int


def create_majorana_string(
    indices: jnp.ndarray,
    coefficient: complex = 1.0,
) -> MajoranaString:
    """
    Create a Majorana string from indices.
    
    Parameters
    ----------
    indices : array
        Binary vector indicating Majorana content
    coefficient : complex
        String coefficient
    
    Returns
    -------
    string : MajoranaString
        Majorana string object
    """
    n_sites = len(indices) // 2
    return MajoranaString(
        indices=jnp.asarray(indices, dtype=jnp.int32),
        coefficient=complex(coefficient),
        n_sites=n_sites,
    )


def compute_string_weight(string: MajoranaString) -> int:
    """
    Compute the weight of a Majorana string.
    
    The weight w_s(v) counts unpaired Majoranas:
        w_s(v) = sum_k |v_{2k-1} - v_{2k}|
    
    Parameters
    ----------
    string : MajoranaString
        Input string
    
    Returns
    -------
    weight : int
        String weight (number of unpaired Majoranas)
    """
    v = string.indices
    n_sites = string.n_sites
    
    # Count unpaired: |v_{2k-1} - v_{2k}| for k = 1, ..., N
    weight = 0
    for k in range(n_sites):
        weight += jnp.abs(v[2*k] - v[2*k + 1])
    
    return int(weight)


def compute_total_weight(string: MajoranaString) -> int:
    """
    Compute total number of Majorana operators in string.
    
    Parameters
    ----------
    string : MajoranaString
        Input string
    
    Returns
    -------
    total : int
        Total weight |v| = sum_i v_i
    """
    return int(jnp.sum(string.indices))


def multiply_strings(
    s1: MajoranaString,
    s2: MajoranaString,
) -> MajoranaString:
    """
    Multiply two Majorana strings.
    
    μ(v1) * μ(v2) = (-1)^{f(v1, v2)} μ(v1 ⊕ v2)
    
    where f counts the sign from anticommutation.
    
    Parameters
    ----------
    s1, s2 : MajoranaString
        Strings to multiply
    
    Returns
    -------
    result : MajoranaString
        Product string
    """
    assert s1.n_sites == s2.n_sites, "Strings must have same number of sites"
    
    v1, v2 = s1.indices, s2.indices
    
    # New indices via XOR
    v_new = jnp.bitwise_xor(v1, v2)
    
    # Compute sign from anticommutation
    # Sign = (-1)^{sum_{i<j} v1_j * v2_i}
    sign = compute_anticommutation_sign(v1, v2)
    
    new_coeff = s1.coefficient * s2.coefficient * sign
    
    return MajoranaString(
        indices=v_new,
        coefficient=new_coeff,
        n_sites=s1.n_sites,
    )


def compute_anticommutation_sign(v1: jnp.ndarray, v2: jnp.ndarray) -> int:
    """
    Compute sign from anticommuting Majorana operators.
    
    Sign = (-1)^{sum_{i<j} v1_j * v2_i}
    """
    n = len(v1)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            count += v1[j] * v2[i]
    return (-1) ** (count % 2)


def string_to_operator(
    string: MajoranaString,
) -> jnp.ndarray:
    """
    Convert Majorana string to matrix representation.
    
    Parameters
    ----------
    string : MajoranaString
        Input string
    
    Returns
    -------
    op : array
        Matrix representation in Fock basis
    """
    n_sites = string.n_sites
    dim = 2 ** n_sites
    
    # Build operator by applying Majorana operators sequentially
    op = string.coefficient * jnp.eye(dim, dtype=complex)
    
    for i, v_i in enumerate(string.indices):
        if v_i == 1:
            gamma_i = majorana_operator(i, n_sites)
            op = op @ gamma_i
    
    return op


def majorana_operator(index: int, n_sites: int) -> jnp.ndarray:
    """
    Construct Majorana operator γ_i in Fock basis.
    
    γ_{2k-1} = c_k + c_k^†
    γ_{2k} = i(c_k - c_k^†)
    
    Parameters
    ----------
    index : int
        Majorana index (0 to 2N-1)
    n_sites : int
        Number of sites N
    
    Returns
    -------
    gamma : array
        Majorana operator matrix
    """
    dim = 2 ** n_sites
    site = index // 2
    is_even = (index % 2 == 0)
    
    # Creation and annihilation operators for site
    c = fermionic_annihilation(site, n_sites)
    c_dag = c.conj().T
    
    if is_even:
        # γ_{2k} = c_k + c_k^†
        gamma = c + c_dag
    else:
        # γ_{2k+1} = i(c_k - c_k^†)
        gamma = 1j * (c - c_dag)
    
    return gamma


def fermionic_annihilation(site: int, n_sites: int) -> jnp.ndarray:
    """
    Construct fermionic annihilation operator c_k.
    
    Includes Jordan-Wigner string for proper anticommutation.
    """
    dim = 2 ** n_sites
    c = jnp.zeros((dim, dim), dtype=complex)
    
    for state in range(dim):
        # Check if site is occupied
        if (state >> site) & 1:
            # Count fermions before this site (Jordan-Wigner)
            n_before = bin(state & ((1 << site) - 1)).count('1')
            sign = (-1) ** n_before
            
            # Annihilate: remove fermion at site
            new_state = state ^ (1 << site)
            c = c.at[new_state, state].set(sign)
    
    return c


def commute_with_hamiltonian(
    string: MajoranaString,
    H: jnp.ndarray,
) -> List[MajoranaString]:
    """
    Compute commutator [H, μ(v)] as sum of Majorana strings.
    
    Parameters
    ----------
    string : MajoranaString
        Input string
    H : array
        Hamiltonian matrix
    
    Returns
    -------
    result : list of MajoranaString
        Strings in the commutator expansion
    """
    # This is a simplified version - full implementation requires
    # expanding H in Majorana basis and computing commutators
    op = string_to_operator(string)
    commutator = H @ op - op @ H
    
    # Decompose commutator back into Majorana strings
    # (This requires solving an inverse problem)
    return decompose_to_strings(commutator, string.n_sites)


def decompose_to_strings(
    operator: jnp.ndarray,
    n_sites: int,
    threshold: float = 1e-10,
) -> List[MajoranaString]:
    """
    Decompose operator into Majorana string basis.
    
    Parameters
    ----------
    operator : array
        Operator matrix
    n_sites : int
        Number of sites
    threshold : float
        Coefficient threshold for inclusion
    
    Returns
    -------
    strings : list of MajoranaString
        Majorana string decomposition
    """
    strings = []
    n_majoranas = 2 * n_sites
    
    # Iterate over all possible strings
    for v in range(2 ** n_majoranas):
        indices = jnp.array([(v >> i) & 1 for i in range(n_majoranas)])
        test_string = create_majorana_string(indices, 1.0)
        test_op = string_to_operator(test_string)
        
        # Coefficient via trace: c = Tr(μ^† O) / Tr(μ^† μ)
        coeff = jnp.trace(test_op.conj().T @ operator) / (2 ** n_sites)
        
        if jnp.abs(coeff) > threshold:
            strings.append(create_majorana_string(indices, coeff))
    
    return strings
