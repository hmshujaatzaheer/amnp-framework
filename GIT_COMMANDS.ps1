# ============================================================
# AMNP Framework - GitHub Repository Setup Commands
# ============================================================
# Run these commands in PowerShell to initialize and push
# your repository to GitHub
# ============================================================

# Step 1: Navigate to your desired directory (modify path as needed)
cd C:\Users\YourUsername\Documents\Projects

# Step 2: Create the repository folder and copy files
# (Assuming you've downloaded/extracted the amnp-framework folder here)
# If using git clone later, skip this step

# Step 3: Initialize Git repository
cd amnp-framework
git init

# Step 4: Configure Git (if not already done globally)
git config user.name "H M Shujaat Zaheer"
git config user.email "shujabis@gmail.com"

# Step 5: Create the repository on GitHub first via web browser:
# 1. Go to https://github.com/new
# 2. Repository name: amnp-framework
# 3. Description: Adaptive Majorana-Neural Propagation for Non-Hermitian Quantum Many-Body Dynamics
# 4. Set to Public
# 5. Do NOT initialize with README (we already have one)
# 6. Click "Create repository"

# Step 6: Add all files to staging
git add .

# Step 7: Create initial commit
git commit -m "Initial commit: AMNP Framework for Non-Hermitian Quantum Many-Body Dynamics

🔬 Adaptive Majorana-Neural Propagation Framework

This repository implements three novel algorithms for simulating 
non-Hermitian quantum many-body dynamics using neural quantum states:

1. GASR (Geometry-Aware Stochastic Reconfiguration)
   - Addresses optimizer incompatibility identified in Hibat-Allah et al. (2025)
   - Adaptively interpolates between SR and Adam based on gradient SNR

2. NHTCT (Non-Hermitian Trotter-Consistent Truncation)  
   - Extends Majorana string propagation to complex eigenvalue spectra
   - Decay-bounded truncation for dissipative systems

3. TENGS (Thermofield-Extended Neural Gibbs States)
   - Finite-temperature states for open quantum systems
   - Non-Hermitian work operator for dissipative dynamics

Features:
- Full JAX/Flax implementation
- NetKet compatibility
- Comprehensive documentation
- Example scripts and tests
- PhD application materials included

Developed as part of PhD application to the Quantum AI Lab, ETH Zürich
Prof. Juan Carrasquilla"

# Step 8: Add remote origin (your GitHub repository)
git remote add origin https://github.com/hmshujaatzaheer/amnp-framework.git

# Step 9: Rename branch to main (if needed)
git branch -M main

# Step 10: Push to GitHub
git push -u origin main

# ============================================================
# Alternative: If you prefer SSH authentication
# ============================================================
# git remote add origin git@github.com:hmshujaatzaheer/amnp-framework.git
# git push -u origin main

# ============================================================
# Verification Commands
# ============================================================

# Check repository status
git status

# View commit history
git log --oneline

# View remote configuration
git remote -v

# ============================================================
# After successful push, your repository will be at:
# https://github.com/hmshujaatzaheer/amnp-framework
# ============================================================

# ============================================================
# Optional: Add topics/tags to repository via GitHub web interface
# Suggested topics:
# - quantum-computing
# - machine-learning
# - neural-networks
# - physics
# - jax
# - variational-methods
# - non-hermitian
# - many-body-physics
# ============================================================
