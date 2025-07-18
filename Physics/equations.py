#import torch as t
#import torch.func as F
import numpy as np
import ufl

""" 
The following two functions create the coefficients for use when solving the pressure gradients. We formulate the equation as:

lap(P_i) = c * P_i + d
"""

def pressure_leading(p: dict):
    """ Computes the coefficient of P_i in the pressure equations """

    coeff = -(p["L_P"] * p["S/V"] + p["L_PL(S/V)_L"]) / p["kappa"]
    
    return coeff

def pressure_constant(p: dict):
    """ Computes the constant term in the pressure equation for Lap(P_i) """

    term1 = p["L_P"] * p["S/V"] * (p["P_b"] - p["sigma_s"] * (p["pi_b"] - p["pi_i"]) )

    term2 = p["L_PL(S/V)_L"] * p["P_L"]

    coeff = (term1 + term2) / p["kappa"]

    return coeff


""" The next equations serve to compute the terms Phi_B and Phi_L in the concentration equations """

def C_P_val(t: float, tau) -> np.ndarray:

    return np.exp(-t / tau)

def comp_phi_B(p: dict, P_i):

    factor = p["L_P"] * p["S/V"]
    term3 = p["sigma_s"] * (p["pi_b"] - p["pi_i"])

    return factor * (p["P_b"] - P_i - term3)

def comp_phi_L(p: dict, P_i):
    return p["L_PL(S/V)_L"] * (P_i - p["P_L"])

def safe_Pe_ratio(Pe, eps=1e-6):
    return (Pe + eps) / (ufl.exp(Pe) - 1.0 + eps)



def comp_Phi_CF(p: dict, P_i):
    phi_B = comp_phi_B(p, P_i)
    phi_L = comp_phi_L(p, P_i)

    Pe = phi_B * (1.0 - p["sigma_f"]) / (p["P"] * p["S/V"])
    ratio = safe_Pe_ratio(Pe)

    Pe_factor = p["P"] * p["S/V"] * ratio * p["tumor"]

    return Pe_factor + phi_L


def comp_Phi_C(p: dict, P_i):
    phi_B = comp_phi_B(p, P_i)
    
    Pe = phi_B * (1.0 - p["sigma_f"]) / (p["P"] * p["S/V"])
    ratio = safe_Pe_ratio(Pe)

    term1 = p["P"] * p["S/V"] * ratio
    term2 = phi_B * (1.0 - p["sigma_f"])

    return p["tumor"] * (term1 + term2)
    


def p_anal(r, p, R):

    alpha_t = R * np.sqrt(p["L_P"]["tumor"] * p["S/V"]["tumor"] / p["kappa"]["tumor"])
    alpha_h = R * np.sqrt(p["L_P"]["normal"] * p["S/V"]["normal"] / p["kappa"]["normal"])
    p_et = p["P_b"] - p["sigma_s"]["tumor"] * (p["pi_b"]["tumor"] - p["pi_i"]["tumor"])
    p_eh = p["P_b"] - p["sigma_s"]["normal"] * (p["pi_b"]["normal"] - p["pi_i"]["normal"])
    p_e = p_eh/p_et

    K = p["kappa"]["tumor"] / p["kappa"]["normal"]

    phi = (1 + alpha_h) * np.sinh(alpha_t)
    theta = K * (alpha_t * np.sinh(alpha_h) - np.sinh(alpha_h))

    P_i = np.empty(r.shape)
    

    P_i[np.where(r <= 1)] = 1 - (1 - p_e) * (alpha_h + 1) * np.sinh(alpha_t * r[np.where(r <= 1)]) / (r[np.where(r <= 1)] * (theta + phi))
    P_i[np.where(r > 1)] = p_e + (1 - p_e) * theta * np.exp(-alpha_h * (r[np.where(r > 1)] - 1)) / (r[np.where(r > 1)] * (theta + phi))
    return p_et * P_i