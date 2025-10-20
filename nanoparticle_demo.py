from Physics.physloss import nano_physics
from ML.model import MLParams


# Enter your desired value for r below:
r = 20e-7 # 1 nm = 1e-7 cm
alpha = 20 # number of drug particles per nanoparticle

# Load nanoparticle parameters
p = MLParams("./Config/nano_physics.json")

# Compute physical properties
sigma_f, P, K_rel, D = nano_physics(r, alpha, p)
print(f"Computed nanoparticle properties for r = {r} cm and alpha = {alpha}:")
print(f"Filtration Coefficient (sigma_f): {sigma_f.item():.4f}")
print(f"Permeability (P) in Tumor: {P.item():.4e}")
print(f"Release Constant (K_rel): {K_rel:.4e}")
print(f"Diffusion Coefficient (D_N) in Tumor: {D:.4e}")
print("Enter these values into your simulation configuration (usually ./Config/sim_params.json) as needed.")