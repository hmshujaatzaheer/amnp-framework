# 🔬 AMNP: Adaptive Majorana-Neural Propagation Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.20+-green.svg)](https://github.com/google/jax)
[![NetKet](https://img.shields.io/badge/NetKet-3.10+-orange.svg)](https://www.netket.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A unified framework for simulating non-Hermitian quantum many-body dynamics using neural quantum states**

## 🎯 Abstract

**AMNP (Adaptive Majorana-Neural Propagation)** is a novel computational framework that unifies autoregressive neural quantum states with Majorana string truncation for simulating **non-Hermitian** strongly correlated fermionic systems. The framework addresses three fundamental gaps in current methods: (1) the optimizer incompatibility between Stochastic Reconfiguration and RNN wave functions, (2) Majorana truncation rules breaking for non-Fock variational states, and (3) restriction of fermionic neural Gibbs states to Hermitian closed systems. We introduce three novel algorithmic contributions: **Geometry-Aware Stochastic Reconfiguration (GASR)**, **Non-Hermitian Trotter-Consistent Truncation (NHTCT)**, and **Thermofield-Extended Neural Gibbs States (TENGS)**. These innovations enable accurate simulation of dissipative Fermi-Hubbard dynamics, non-Hermitian kagome lattices, and open-system quantum transport beyond existing classical methods.

---

### 📦 Repository Contents | GitHub: [hmshujaatzaheer/amnp-framework](https://github.com/hmshujaatzaheer/amnp-framework)

| Component | Description | Location |
|-----------|-------------|----------|
| **GASR** | Geometry-Aware Stochastic Reconfiguration optimizer | `amnp/optimizers/` |
| **NHTCT** | Non-Hermitian Trotter-Consistent Truncation | `amnp/truncation/` |
| **TENGS** | Thermofield-Extended Neural Gibbs States | `amnp/neural_states/` |
| **Hamiltonians** | Fermi-Hubbard (Hermitian & Dissipative), Heisenberg, Kagome | `amnp/utils/` |
| **Examples** | Demonstration scripts for each algorithm | `examples/` |
| **Tests** | Unit tests for GASR optimizer | `tests/` |
| **Theory** | Mathematical foundations document | `docs/theory.md` |
| **PhD Application** | Letter of Motivation, Research Statement, Research Proposal | `application_materials/` |

**Keywords:** `neural-quantum-states` `non-hermitian` `variational-monte-carlo` `fermionic-systems` `machine-learning` `quantum-many-body` `jax` `stochastic-reconfiguration` `majorana-operators` `thermal-states`

---

## 🔥 Motivation: Addressing Critical Research Gaps

This framework addresses three fundamental limitations identified in recent literature:

### Gap 1: Optimizer Incompatibility (Hibat-Allah et al., 2025)
> *"Stochastic Reconfiguration is not as effective as Adam optimizer when applied to RNN wave functions"*

**Our Solution:** GASR adaptively interpolates between curvature-aware (SR) and first-order (Adam) updates based on gradient signal-to-noise ratio.

### Gap 2: Majorana Truncation for Non-Hermitian Systems (D'Anna, Nys & Carrasquilla, 2025)
> Truncation rules assume Fock-state initial conditions, breaking for variational states

**Our Solution:** NHTCT introduces decay-bounded truncation tied to complex eigenvalue spectra.

### Gap 3: Hermitian Restriction in Neural Thermal States (Nys & Carrasquilla, 2025)
> *"Extensions to real-time dynamics and non-Hermitian steady states require new theoretical frameworks"*

**Our Solution:** TENGS extends thermofield-double purification to dissipative fermionic dynamics.

## 📁 Repository Structure

```
amnp-framework/
├── amnp/                          # Core library
│   ├── __init__.py
│   ├── optimizers/                # GASR implementation
│   │   ├── gasr.py               # Geometry-Aware SR optimizer
│   │   ├── quantum_geometric_tensor.py
│   │   └── snr_estimator.py      # Signal-to-noise ratio estimation
│   ├── truncation/                # NHTCT implementation
│   │   ├── nhtct.py              # Non-Hermitian truncation
│   │   ├── majorana_strings.py   # Majorana string algebra
│   │   └── decay_bounds.py       # Decay-bounded pruning
│   ├── neural_states/             # TENGS implementation
│   │   ├── tengs.py              # Thermofield-Extended Neural Gibbs
│   │   ├── bivit_backflow.py     # BiViT attention + backflow
│   │   ├── rnn_wavefunction.py   # Autoregressive ansatz
│   │   └── jastrow.py            # Jastrow correlation factor
│   └── utils/                     # Utilities
│       ├── hamiltonians.py       # Fermi-Hubbard, Heisenberg, kagome
│       ├── observables.py        # Energy, correlations, v-scores
│       └── visualization.py      # Plotting utilities
├── examples/                      # Usage examples
│   ├── 01_gasr_optimization.py
│   ├── 02_nhtct_truncation.py
│   └── 03_tengs_thermal_states.py
├── tests/                         # Unit tests
│   └── test_gasr.py
├── docs/                          # Documentation
│   └── theory.md                 # Mathematical foundations
├── application_materials/         # PhD application documents
│   ├── Letter_of_Motivation.md
│   ├── Statement_of_Research_Interests.md
│   ├── Research_Proposal.pdf
│   └── Research_Proposal.tex
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## 🚀 Installation

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (optional, for GPU acceleration)

### Install from source
```bash
git clone https://github.com/hmshujaatzaheer/amnp-framework.git
cd amnp-framework
pip install -e .
```

### Install dependencies only
```bash
pip install -r requirements.txt
```

## 💡 Quick Start

### 1. GASR Optimizer for RNN Wave Functions

```python
import jax.numpy as jnp
from amnp.optimizers import GASR
from amnp.neural_states import RNNWavefunction
from amnp.utils import FermiHubbardHamiltonian

# Define system
L = 4  # 4x4 lattice
hamiltonian = FermiHubbardHamiltonian(Lx=L, Ly=L, U=4.0, t=1.0)

# Initialize RNN ansatz
model = RNNWavefunction(
    hidden_size=64,
    num_layers=2,
    num_sites=L*L
)

# Create GASR optimizer with adaptive interpolation
optimizer = GASR(
    learning_rate=1e-3,
    snr_threshold=10.0,  # τ parameter
    regularization=1e-4   # λ parameter
)
```

### 2. NHTCT for Non-Hermitian Dynamics

```python
from amnp.truncation import NHTCT
from amnp.utils import DissipativeFermiHubbard

# Non-Hermitian Hamiltonian with particle loss
H = DissipativeFermiHubbard(
    Lx=4, Ly=4, U=4.0, t=1.0,
    gamma=0.1  # Dissipation rate
)

# Initialize truncation scheme
truncation = NHTCT(
    max_weight=6,           # S parameter
    trotter_error=1e-4,     # ε_Trotter
    time_step=0.01          # δτ
)

# Evolve under non-Hermitian dynamics
times, states = truncation.evolve(H.to_matrix(), psi_0, t_final=10.0)
```

### 3. TENGS for Finite-Temperature Non-Hermitian Systems

```python
from amnp.neural_states import TENGS, NonHermitianWorkOperator

# Create TENGS model
tengs = TENGS(config=TENGSConfig(
    n_sites=4,
    beta_0=0.1,  # Initial inverse temperature
    beta=1.0,    # Target inverse temperature
    hidden_size=128
))

# Construct non-Hermitian work operator
W_NH = NonHermitianWorkOperator(
    H=H_physical,
    H_0=H_reference,
    gamma=dissipator,
    beta_0=0.1, beta=1.0
)
```

## 🧮 Mathematical Foundations

### GASR: Adaptive Quantum Geometric Tensor

The GASR optimizer uses an interpolated metric:

$$G_{\text{GASR}} = (1-\alpha)S + \alpha J^T J + \lambda I$$

where:
- $S_{kk'} = \langle \Delta O_k^* \Delta O_{k'} \rangle$ is the quantum geometric tensor
- $J$ is the Jacobian of log-amplitudes
- $\alpha = \sigma(\log \rho - \tau)$ adapts based on SNR $\rho = \|g\|^2 / \text{Var}[g]$

### NHTCT: Decay-Bounded Truncation

For non-Hermitian $H = H_{\text{Herm}} + i\Gamma$:

$$\text{Truncate if: } w_s(v) > S \text{ OR } |\text{Im}[\lambda_v]| > \Gamma_{\max}$$

where $\Gamma_{\max} = -\frac{1}{\delta\tau}\ln(\epsilon_{\text{Trotter}})$

### TENGS: Non-Hermitian Work Operator

$$W_{\text{NH}} = \hat{H} \otimes \tilde{I} - \frac{\beta_0}{\beta} \hat{I} \otimes \tilde{H}_0 + i\Gamma \otimes \tilde{I}$$

## 🎯 Target Systems

As outlined in the research proposal, the AMNP framework targets:

| System | AMNP | MPS/fPEPS | QMC |
|--------|------|-----------|-----|
| Dissipative Fermi-Hubbard | ✓ Proposed | Limited | Sign problem |
| Non-Hermitian kagome | ✓ Proposed | ✗ | ✗ |
| Open-system transport | ✓ Proposed | 1D only | ✗ |
| Finite-T non-Hermitian | ✓ Proposed | ✗ | ✗ |

**Validation Protocol:** Following variational benchmarks (Wu et al., Science 2024), v-scores will compare energy accuracy against exact diagonalization and QMC where applicable.

## 📚 References

1. D. Wu et al., "Variational benchmarks for quantum many-body problems," *Science* **386**, 296-301 (2024)
2. M. Hibat-Allah et al., "Recurrent neural network wave functions for Rydberg atom arrays on kagome lattice," *Commun. Phys.* **8**, 308 (2025)
3. M. D'Anna, J. Nys, J. Carrasquilla, "Majorana string simulation of nonequilibrium dynamics in two-dimensional lattice fermion systems," *arXiv:2511.02809* (2025)
4. J. Nys, J. Carrasquilla, "Fermionic neural Gibbs states," *arXiv:2512.04663* (2025)
5. G. Carleo, M. Troyer, "Solving the quantum many-body problem with artificial neural networks," *Science* **355**, 602-606 (2017)

## 👤 Author

**H M Shujaat Zaheer**
- MSc Computer Science (AI/ML Specialization), University of Sialkot
- Quantum Information Certification, KAIST
- Deep Learning Certification, IBM

📧 Email: shujabis@gmail.com  
🔗 GitHub: [@hmshujaatzaheer](https://github.com/hmshujaatzaheer)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work is inspired by and builds upon the pioneering research of Prof. Juan Carrasquilla and collaborators at the Quantum AI Lab, ETH Zürich. The framework addresses specific research gaps identified in their 2025 publications.

---

<p align="center">
  <i>Developed as part of PhD application to the Quantum AI Lab, ETH Zürich</i><br>
  <i>Supervisor: Prof. Juan Carrasquilla</i>
</p>
