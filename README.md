# NP-PINN

This project addresses both the forward and inverse modeling challenges associated with nanoparticle-based drug delivery in solid tumors.

## Mathematical Model

We use a model for drug delivery consisting of:

1. Conservation equations governing interstitial fluid dynamics  
2. Mass transport equations for drug-loaded nanoparticles  
3. Mass transport equations for free and internalized drug  
4. Equations modeling effects arising from nanoparticle size  

In this model we consider exchange between plasma and interstitium, constant-rate drug release, and cellular internalization. Spatial dynamics are included for nanoparticles and free drug, while internalized drug remains fixed. Parameters arise from tissue properties, drug properties (DOX), and nanoparticle design.

### Interstitial Fluid Dynamics

Fluid dynamics within the tumor follow Darcy’s law:

v_i = -κ ∇P_i

where \(v_i\), \(κ\), and \(P_i\) denote interstitial fluid velocity, hydraulic conductivity, and interstitial fluid pressure.

The source and sink terms are modeled as:

∇·v_i = φ_B − φ_L

with

φ_B = L_P (S/V) (P_B − P_i − σ_s(π_b − π_i))
φ_L = L_PL (S/V)_L (P_i − P_L)

### Drug Mass Transport Equations

Drug movement is influenced by advection (from interstitial flow) and diffusion (Fick’s law), along with reaction terms for release, internalization, and decay. This yields a system of ADR equations:

C_p = exp(−t/τ)

∂C_N/∂t = −∇·(v_i C_N)
+ ∇·(D_N ∇C_N)
+ (Φ_B − Φ_L)
− K_rel C_N

∂C_F/∂t = −∇·(v_i C_F)
+ ∇·(D_F ∇C_F)
+ α K_rel C_N − K_INT C_F
+ K_DEG-INT C_INT − K_DEG-F C_F

∂C_INT/∂t = K_INT C_F − K_DEG-INT C_INT

Here \(C_P, C_N\) denote plasma and interstitial nanoparticle concentrations; \(C_F, C_{INT}\) denote free and internalized drug. Reaction constants \(K_{INT}, K_{DEG-F}, K_{DEG-INT}\) are known.

Nanoparticle-dependent parameters include:

- \(α\): number of drug molecules per nanoparticle  
- \(τ\): nanoparticle half-life  
- \(k_{rel}\): drug release rate  
- \(D_N\): nanoparticle diffusion constant  
- \(Φ_L, Φ_B\): lymphatic and blood-derived flows  

All are derived from three design parameters: \(α\), \(τ\), and diameter \(d\).

### Nanoparticle Physics

Define the ratio of nanoparticle diameter to pore size:

λ = d / d₀

Hydrostatic and electrostatic interaction terms:

H = 6πF / K_t
W = F(2 − F)K_s / (2K_t)

where \(F = (1 − λ)^2\), with empirical parameters \(K_t, K_s\):

[K_t, K_s]^T =
(9/4) π² √2 (1−λ)^(−5/2)
[1 + Σ_{n=1}^2 (a_n, b_n)^T (1−λ)^n]

Σ_{n=0}^4 (a_{n+3}, b_{n+3})^T λ^n

Diffusion in free solution by Stokes–Einstein:

D₀ = k_B T / (6π η d)

Corrected for ECM hindrance:

D / D₀ = exp(−a σ^b) · exp(−0.84 f^{1.09})

with \(f = (1 + λ’)(2σ)\).

Vessel permeability and reflection coefficient:

P = γ H D₀ / L
σ_f = 1 − W

### Nanoparticle Fluid Flow

Péclet number:

Pe = φ_B (1 − σ_f) / (P S/V)

Blood-derived source term:

Φ_B = φ_B(1 − σ_f) C_P
+ P(S/V)(C_P − C_N)(Pe / (1 + e^{Pe}))

Lymphatic sink:

Φ_L = φ_L C_N

## Simulation

Simulations use FEniCSx. The computational domain is a 4×4 cm grid with 200 µm spacing, refined to 100 µm near tumor boundaries. Continuous Galerkin elements (degree 1) are used, with solver tolerance \(10^{-8}\). Time discretization uses implicit Euler with \(Δt = 0.1\,s\).

### Boundary Conditions

External boundary:  
- Open boundary for fluid  
- Dirichlet for concentration  

Tumor boundary:  
- Continuity of flow and concentration  
- Bicubic interpolation for differentiability where needed  

## Machine Learning

FEM simulation is computationally costly; we therefore use physics-informed neural networks (PINNs).

### Model Architecture

A PINN minimizes both data loss and PDE residual:

L_PINN = MSE + λ L_PDE

We use deep MLPs with SiLU activation, six hidden layers of width 1024.

### Data Generation

Training data come from FEM simulations. Only initial conditions and two full-state snapshots are provided. Inputs are min-max scaled.

### Training Procedure

For IFP/IFV: input is 2D spatial coordinates; output is IFP.  
For drug transport: input is (x, y, t); output is (C_N, C_P, C_INT).

Optimization:

1. ADAM for 1000 epochs, starting at learning rate \(10^{-4}\)  
2. L-BFGS for 500 epochs, learning rate 0.1  

Performance is evaluated on full FEM simulations.

