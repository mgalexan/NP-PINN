import torch as t
import math as m

def gradient(tens, coords, type= "temporal"):
    
    grad = t.autograd.grad(
        outputs=tens,
        inputs=coords,
        grad_outputs=t.ones_like(tens),
        create_graph=True,
        retain_graph=True,
    )[0]  # shape (N, d)
    if type == "temporal":
        return grad[:, 1:]  # keep only x, y components
    elif type == "spatial":
        return grad

def gradient_radial(tens, coords):
    """
    Compute radial gradient in spherical coordinates.
    coords has shape (N, 2): [t, r]
    Returns dr component only, shape (N, 1)
    """
    grad = t.autograd.grad(
        outputs=tens,
        inputs=coords,
        grad_outputs=t.ones_like(tens),
        create_graph=True,
        retain_graph=True,
    )[0]  # shape (N, 2)
    return grad[:, 1:2]  # keep only r component, shape (N, 1)

    
def divergence(field, coords, type= "temporal"):
    
    div = t.zeros(field.shape[0], device=coords.device)
    for j in range(2):  # field.shape[1] == 2
        assert field.requires_grad
        grad = t.autograd.grad(
            outputs=field[:, j],
            inputs=coords,
            grad_outputs=t.ones_like(field[:, j]),
            create_graph=True,
            retain_graph=True  # Needed since we're doing multiple grad calls
        )[0]  # shape: (N, 3)
        if type == "temporal":
            div += grad[:, j + 1]  # Skip time (dim 0), use x=1, y=2
        else:
            div += grad[:,j]
    
    
    return div.unsqueeze(-1)  # shape: (N,)

def divergence_radial(field, coords):
    """
    Compute divergence of a radial vector field in spherical coordinates.
    field has shape (N, 1): radial component only
    coords has shape (N, 2): [t, r]
    Returns div shape (N, 1)
    
    In spherical coordinates: div(v) = (1/r²) d(r² * v_r)/dr
    """
    r = coords[:, 1:2]  # shape (N, 1)
    
    # Compute d(r² * field)/dr
    r_squared_field = r**2 * field
    
    grad = t.autograd.grad(
        outputs=r_squared_field,
        inputs=coords,
        grad_outputs=t.ones_like(r_squared_field),
        create_graph=True,
        retain_graph=True,
    )[0]  # shape (N, 2)
    
    d_r_squared_field = grad[:, 1:2]  # dr component, shape (N, 1)
    
    # Divide by r²
    div = d_r_squared_field / (r**2 + 1e-10)  # add small epsilon to avoid division by zero
    
    return div

def laplacian_radial(tens, coords):
    """
    Compute Laplacian in spherical coordinates with radial symmetry.
    tens has shape (N, 1)
    coords has shape (N, 2): [t, r]
    Returns Laplacian shape (N, 1)
    
    In spherical coordinates: ∇²u = d²u/dr² + (2/r) du/dr
    """
    r = coords[:, 1:2]  # shape (N, 1)
    
    # First derivative
    grad_u = t.autograd.grad(
        outputs=tens,
        inputs=coords,
        grad_outputs=t.ones_like(tens),
        create_graph=True,
        retain_graph=True,
    )[0]  # shape (N, 2)
    
    du_dr = grad_u[:, 1:2]  # shape (N, 1)
    
    # Second derivative
    grad2_u = t.autograd.grad(
        outputs=du_dr,
        inputs=coords,
        grad_outputs=t.ones_like(du_dr),
        create_graph=True,
        retain_graph=True,
    )[0]  # shape (N, 2)
    
    d2u_dr2 = grad2_u[:, 1:2]  # shape (N, 1)
    
    # Combine: ∇²u = d²u/dr² + (2/r) du/dr
    laplacian = d2u_dr2 + (2.0 / (r + 1e-10)) * du_dr
    
    return laplacian

def diff_t(tens, coords):
    grad = t.autograd.grad(
        outputs=tens,
        inputs=coords,
        grad_outputs=t.ones_like(tens),
        create_graph=True,
        retain_graph=True,
    )[0]
    return grad[:,0]


def compute_phi_B(P_i, coords, p):
    L_p = p["L_P"](coords)
    SV = p["S/V"](coords)
    P_b = p["P_b"](coords)
    sigma_s = p["sigma_s"](coords)
    pi_b = p["pi_b"](coords)
    pi_i = p["pi_i"](coords)
    
    
    return L_p * SV * (P_b - P_i - sigma_s * (pi_b - pi_i))

def compute_phi_L(P_i, coords, p):
    L_pl = p["L_PL(S/V)_L"](coords)
    P_L = p["P_L"](coords)
    

    return L_pl * (P_i - P_L)

def pressure_phys_loss(P_i, coords, p):
    lhs = divergence(gradient(P_i, coords, "spatial"), coords, "spatial") * p["kappa"](coords)
    rhs = compute_phi_B(P_i, coords, p) - compute_phi_L(P_i, coords, p)
    return t.sqrt(t.sum(t.square(lhs + rhs)))

def pressure_phys_loss_radial(P_i, coords, p):
    """
    Pressure physics loss in spherical coordinates with radial symmetry.
    coords: (N, 2) tensor [t, r]
    P_i: (N, 1) interstitial pressure
    p: dictionary with parameter functions
    
    PDE: kappa * ∇²P_i = Phi_B - Phi_L
    where ∇² is the Laplacian in spherical coordinates
    """
    lhs = laplacian_radial(P_i, coords) * p["kappa"](coords)
    rhs = compute_phi_B(P_i, coords, p) - compute_phi_L(P_i, coords, p)
    return t.mean(t.square(lhs + rhs))

def compute_Pe_ratio(SV, P, sigma_f, phi_B):


    Pe = phi_B * (1 - sigma_f) / (P * SV)
    
    epsilon = 1e-6

    Pe_ratio = (Pe + epsilon) / (t.exp(Pe) + epsilon - 1)
    return Pe_ratio

def compute_C_p(coords, tau):
    times = coords[:,0].unsqueeze(-1)
    C_p = t.exp(-times / tau)
    return C_p

def compute_Phi_C(P, sigma_f, tau, Pe_ratio, phi_B, SV, tumor, coords):

    C_p = compute_C_p(coords, tau)
    term1 = phi_B * (1 - sigma_f)
    term2 = P * SV * Pe_ratio
    return C_p * (term1 + term2) * tumor

def compute_Phi_N(P, Pe_ratio, phi_L, SV, tumor, coords, p):

    return P * SV * Pe_ratio * tumor + phi_L
    

def C_N_Loss(coords, C_N, D_N, v_i, v_i_div, K_rel, Phi_C, Phi_N):

    
    lhs = diff_t(C_N, coords)

    rhs =  D_N * divergence(gradient(C_N, coords), coords) 

    rhs -= t.sum(gradient( C_N, coords) * v_i, dim= -1).unsqueeze(-1)
    rhs -= v_i_div * C_N
    
    rhs += -K_rel * C_N + Phi_C - Phi_N * C_N

    return (t.mean(t.square(lhs - rhs)))

def C_F_Loss(coords, C_F, C_N, C_INT, D_F, v_i, v_i_div, alpha, K_rel, K_INT, K_degINT, K_degF):

    lhs = diff_t(C_F, coords)

    rhs = D_F * divergence(gradient(C_F, coords), coords)
    rhs -= t.sum(gradient( C_F, coords) * v_i, dim= -1).unsqueeze(-1)
    rhs -= v_i_div * C_F
    rhs += alpha * K_rel * C_N - K_INT * C_F
    rhs += K_degINT * C_INT - K_degF * C_F

    return (t.mean(t.square(lhs - rhs)))

def C_INT_Loss(coords, C_INT, C_F, K_degINT, K_INT):

    lhs = diff_t(C_INT, coords)

    rhs = K_INT * C_F - K_degINT * C_INT

    return t.mean(t.square(lhs - rhs))

def N_Loss(coords, N, rho, K, D):

    lhs = diff_t(N, coords)

    rhs = divergence(D * gradient(N, coords), coords)

    rhs += rho * N * (1 - N / K)

    return t.mean(t.square(lhs - rhs))


def nano_physics(d, alpha, p, device= t.device("cpu")):

    # Compute lambda

    lam = d / p["d_0"]

    exps = t.arange(0, 5).to(device)       # powers 0–4
    exps_minus = t.arange(1, 3).to(device) # powers 1–2

    lam_exp = lam ** exps
    minus_lam_exp = (1 - lam) ** exps_minus

    a_vals = t.tensor([p["a_1"], p["a_2"], p["a_3"], p["a_4"], p["a_5"], p["a_6"], p["a_7"]]).to(device)
    b_vals = t.tensor([p["b_1"], p["b_2"], p["b_3"], p["b_4"], p["b_5"], p["b_6"], p["b_7"]]).to(device)

    leading_lambda = (9/4) * (t.pi**2) * t.sqrt(t.tensor(2.0)) * (1 - lam) ** (-2.5)

    K_t = leading_lambda * (1 + t.dot(a_vals[0:2], minus_lam_exp)) + t.dot(a_vals[2:], lam_exp)
    K_s = leading_lambda * (1 + t.dot(b_vals[0:2], minus_lam_exp)) + t.dot(b_vals[2:], lam_exp)


    F = minus_lam_exp[1]

    # Compute W and H

    W = F * (2 - F) * K_s / (2 * K_t)
    H = 6 * m.pi * F / K_t

    # Compute D_0

    D_0 = p["K_b"] * p["T"] / (3 * m.pi * p["eta"] * d)



    # Compute a, b, sigma for D:
    lambda_prime = d / p["d_f"]
    b = 0.174 * m.log(59.6 / lambda_prime)
    sigma = p["sigma"]  
    f = (1 + lambda_prime) / (2 * sigma)




    # Now compute the parameters we need to return

    sigma_f = 1 - W

    P = p["gamma"] * H * D_0 / p["L"]

    K_rel = 5.04e-5 / (alpha * (1 + d/ 100e-7))

    D = D_0 * m.exp( - p["a"] * sigma ** b - 0.84 * f ** 1.09)

    

    return sigma_f, P, K_rel, D

def C_N_loss_radial(coords, C_N, D_N, v_r, div_v_r, K_rel, Phi_C, Phi_N):
    """
    Nanoparticle concentration loss in spherical coordinates with radial symmetry.
    coords: (N, 2) tensor [t, r]
    C_N: (N, 1) concentration
    D_N: (N, 1) diffusivity
    v_r: (N, 1) radial velocity
    div_v_r: (N, 1) divergence of radial velocity
    K_rel: (N, 1) release rate
    Phi_C, Phi_N: (N, 1) source terms
    
    PDE: ∂C_N/∂t = D_N * ∇²C_N - div(v_r * C_N) - K_rel*C_N + Phi_C - Phi_N*C_N
    where div(v_r * C_N) = v_r * ∂C_N/∂r + C_N * div(v_r)
    """
    lhs = diff_t(C_N, coords)
    
    # Diffusion term: D_N * ∇²C_N
    rhs = D_N * laplacian_radial(C_N, coords)
    
    # Convection term: -div(v_r * C_N) = -(v_r * ∂C_N/∂r + C_N * div(v_r))
    grad_C_N = gradient_radial(C_N, coords)
    rhs -= v_r * grad_C_N
    rhs -= C_N * div_v_r
    
    # Reaction terms
    rhs += -K_rel * C_N + Phi_C - Phi_N * C_N
    
    return t.mean(t.square(lhs - rhs))


def C_F_loss_radial(coords, C_F, C_N, C_INT, D_F, v_r, div_v_r, alpha, K_rel, K_INT, K_degINT, K_degF):
    """
    Free concentration loss in spherical coordinates with radial symmetry.
    coords: (N, 2) tensor [t, r]
    C_F: (N, 1) free concentration
    v_r: (N, 1) radial velocity
    div_v_r: (N, 1) divergence of radial velocity
    """
    lhs = diff_t(C_F, coords)
    
    # Diffusion term
    rhs = D_F * laplacian_radial(C_F, coords)
    
    # Convection term: -div(v_r * C_F) = -(v_r * ∂C_F/∂r + C_F * div(v_r))
    grad_C_F = gradient_radial(C_F, coords)
    rhs -= v_r * grad_C_F
    rhs -= C_F * div_v_r
    
    # Reaction terms
    rhs += alpha * K_rel * C_N - K_INT * C_F
    rhs += K_degINT * C_INT - K_degF * C_F
    
    return t.mean(t.square(lhs - rhs))


def C_INT_loss_radial(coords, C_INT, C_F, K_degINT, K_INT):
    """
    Internalized concentration loss in spherical coordinates with radial symmetry.
    coords: (N, 2) tensor [t, r]
    C_INT: (N, 1) internalized concentration
    No spatial diffusion, only reaction terms.
    """
    lhs = diff_t(C_INT, coords)
    
    rhs = K_INT * C_F - K_degINT * C_INT
    
    return t.mean(t.square(lhs - rhs))


