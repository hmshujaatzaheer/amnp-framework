# AMNP Framework: Theoretical Foundations

## Overview

The **Adaptive Majorana-Neural Propagation (AMNP)** framework introduces three novel algorithms for simulating non-Hermitian quantum many-body dynamics:

1. **GASR**: Geometry-Aware Stochastic Reconfiguration
2. **NHTCT**: Non-Hermitian Trotter-Consistent Truncation  
3. **TENGS**: Thermofield-Extended Neural Gibbs States

---

## 1. GASR: Geometry-Aware Stochastic Reconfiguration

### Motivation

Standard Stochastic Reconfiguration (SR) uses the quantum geometric tensor (QGT) for natural gradient descent:

$$\delta\theta = -\eta S^{-1} f$$

where $S_{kk'} = \langle \partial_k \log\psi^* \partial_{k'} \log\psi \rangle - \langle \partial_k \log\psi^* \rangle \langle \partial_{k'} \log\psi \rangle$

**Problem**: Hibat-Allah et al. (2025) identified that *"Stochastic Reconfiguration is not as effective as Adam optimizer when applied to RNN wave functions"* due to high-variance QGT estimates in autoregressive architectures.

### Solution: Adaptive Interpolation

GASR uses an interpolated metric that adapts based on gradient signal-to-noise ratio (SNR):

$$G_{\text{GASR}} = (1-\alpha)S + \alpha J^T J + \lambda I$$

where:
- $S$ is the quantum geometric tensor
- $J$ is the Jacobian of log-amplitudes
- $\alpha = \sigma(\log \rho - \tau)$ with SNR $\rho = \|g\|^2 / \text{Var}[g]$
- $\lambda$ is regularization

### Algorithm

```
Input: parameters θ, samples {x}, learning rate η, threshold τ
1. Compute gradient g = ∇_θ E
2. Estimate SNR: ρ = ||g||² / Var[g]
3. Compute α = σ(log ρ - τ)
4. Compute QGT S from samples
5. Compute Jacobian term J^T J
6. Form G_GASR = (1-α)S + αJ^T J + λI
7. Solve G_GASR δθ = -g
8. Update θ ← θ + η·δθ
Output: updated parameters θ
```

### Properties

- **High SNR** (ρ >> τ): α → 0, recovers SR with curvature information
- **Low SNR** (ρ << τ): α → 1, reduces to preconditioned gradient descent
- Smoothly interpolates between regimes

---

## 2. NHTCT: Non-Hermitian Trotter-Consistent Truncation

### Motivation

Majorana string propagation (D'Anna et al., 2025) truncates strings based on weight:

$$\text{Truncate if } w_s(v) > S$$

where $w_s(v)$ counts unpaired Majoranas.

**Problem**: This assumes Fock-state initial conditions and breaks for:
- Variational initial states
- Non-Hermitian Hamiltonians with complex eigenvalues

### Solution: Decay-Bounded Truncation

For non-Hermitian $H = H_{\text{Herm}} + i\Gamma$, eigenvalues are complex: $\lambda = E - i\gamma$

NHTCT introduces dual truncation criteria:

$$\text{Truncate if: } w_s(v) > S \text{ OR } |\text{Im}[\lambda_v]| > \Gamma_{\max}$$

where the decay bound is:

$$\Gamma_{\max} = -\frac{1}{\delta\tau}\ln(\epsilon_{\text{Trotter}})$$

### Derivation

For a Trotter step $e^{-iH\delta\tau}$, the error from truncating a mode with decay rate $\gamma$ is:

$$\epsilon \sim e^{-\gamma \delta\tau}$$

Setting $\epsilon = \epsilon_{\text{Trotter}}$ and solving:

$$\gamma_{\max} = -\frac{\ln(\epsilon_{\text{Trotter}})}{\delta\tau}$$

### Algorithm

```
Input: Majorana strings {μ(v)}, eigenvalues {λ_v}, S, δτ, ε
1. Compute Γ_max = -ln(ε)/δτ
2. For each string μ(v):
   a. Compute weight w_s(v)
   b. Extract decay rate γ_v = |Im[λ_v]|
   c. If w_s(v) > S OR γ_v > Γ_max:
      Truncate string
3. Propagate remaining strings
Output: Truncated string set
```

---

## 3. TENGS: Thermofield-Extended Neural Gibbs States

### Motivation

Neural Gibbs states (Nys & Carrasquilla, 2025) use thermofield double purification:

$$|\Psi(\beta)\rangle = e^{-\beta W/2}|\Psi_0(\beta_0)\rangle$$

with work operator $W = H \otimes \tilde{I} - \frac{\beta_0}{\beta} \hat{I} \otimes \tilde{H}_0$

**Problem**: *"Extensions to real-time dynamics and non-Hermitian steady states require new theoretical frameworks"*

### Solution: Non-Hermitian Work Operator

TENGS extends to dissipative systems via:

$$W_{\text{NH}} = \hat{H} \otimes \tilde{I} - \frac{\beta_0}{\beta} \hat{I} \otimes \tilde{H}_0 + i\Gamma \otimes \tilde{I}$$

where $\Gamma$ is the dissipator (e.g., particle loss $\Gamma = \gamma \sum_i n_i$).

### Imaginary-Time Evolution

The thermal state is obtained via projected imaginary-time evolution (tre-pITE):

$$|\Psi(\tau + \delta\tau)\rangle = \frac{e^{-W_{\text{NH}}\delta\tau}|\Psi(\tau)\rangle}{\|e^{-W_{\text{NH}}\delta\tau}|\Psi(\tau)\rangle\|}$$

### Neural Architecture

TENGS uses BiViT (Bidirectional Vision Transformer) with backflow:

```
Input: x_physical, x_auxiliary
    ↓
[2D-GRU Embedding]
    ↓
[BiViT Attention Blocks] ← Cross-attention between physical/auxiliary
    ↓
[Backflow Transformation]
    ↓
[Slater Determinant + Jastrow]
    ↓
Output: log Ψ(x, x̃)
```

### Properties

- Handles non-Hermitian steady states
- Captures physical-auxiliary entanglement
- Compatible with GASR optimizer
- Scales to 2D systems

---

## Target Systems

### 1. Dissipative Fermi-Hubbard Model

$$H = -t\sum_{\langle ij\rangle\sigma}(c^\dagger_{i\sigma}c_{j\sigma} + \text{h.c.}) + U\sum_i n_{i\uparrow}n_{i\downarrow} + i\gamma\sum_i n_i$$

### 2. Non-Hermitian Kagome Lattice

Rydberg atom arrays with complex detuning for frustrated magnetism.

### 3. Open-System Transport

Boundary-driven fermions at finite temperature with particle injection/removal.

---

## Validation Protocol

Following Wu et al., Science 386, 296 (2024):

1. **v-score**: $v = -\log_{10}(|E - E_{\text{exact}}|/|E_{\text{exact}}| + \sigma_E^2/E_{\text{exact}}^2)$
2. **Correlation functions**: Spin-spin, density-density
3. **Spectral properties**: Lindbladian gap estimation

---

## References

1. D. Wu et al., "Variational benchmarks for quantum many-body problems," Science 386, 296-301 (2024)
2. M. Hibat-Allah et al., "Recurrent neural network wave functions for Rydberg atom arrays on kagome lattice," Commun. Phys. 8, 308 (2025)
3. M. D'Anna, J. Nys, J. Carrasquilla, "Majorana string simulation of nonequilibrium dynamics," arXiv:2511.02809 (2025)
4. J. Nys, J. Carrasquilla, "Fermionic neural Gibbs states," arXiv:2512.04663 (2025)
