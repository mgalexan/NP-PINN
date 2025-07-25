import torch as t

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

    
def divergence(field, coords, type= "temporal"):
    
    div = t.zeros(field.shape[0], device=coords.device)
    for j in range(2):  # field.shape[1] == 2
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

    return t.sqrt(t.sum(t.square(lhs - rhs)))

def C_F_Loss(coords, C_F, C_N, C_INT, D_F, v_i, v_i_div, alpha, K_rel, K_INT, K_degINT, K_degF):

    lhs = diff_t(C_F, coords)

    rhs = D_F * divergence(gradient(C_F, coords), coords)
    rhs -= t.sum(gradient( C_F, coords) * v_i, dim= -1).unsqueeze(-1)
    rhs -= v_i_div * C_F
    rhs += alpha * K_rel * C_N - K_INT * C_F
    rhs += K_degINT * C_INT - K_degF * C_F

    return t.sqrt(t.sum(t.square(lhs - rhs)))

def C_INT_Loss(coords, C_INT, C_F, K_degINT, K_INT):

    lhs = diff_t(C_INT, coords)

    rhs = K_INT * C_F - K_degINT * C_INT

    return t.sqrt(t.sum(t.square(lhs - rhs)))








    



