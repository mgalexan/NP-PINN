import torch as t
import torch.func as F
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

def comp_Phi_CF(p: dict, P_i):

    phi_B = comp_phi_B(p, P_i)

    phi_L = comp_phi_L(p, P_i)

    eps = 1e-8

    Pe = phi_B * (1.0 - p["sigma_f"]) / (p["P"] * p["S/V"])

    Pe_factor =  p["P"] * p["S/V"] * Pe / (ufl.exp(Pe) - 1.0 + eps)

    return ufl.max_value(ufl.min_value(Pe_factor + phi_L, 1e3), 0.0)

def comp_Phi_C(p: dict, P_i):

    phi_B = comp_phi_B(p, P_i)

    eps = 1e-8
    
    Pe = phi_B * (1.0 - p["sigma_f"]) / (p["P"] * p["S/V"])

    term1 = p["P"] * p["S/V"] * Pe / (ufl.exp(Pe) - 1.0 + eps)


    term2 = phi_B * (1.0 - p["sigma_f"]) 

    return ufl.max_value(ufl.min_value(term1 + term2, 1e3), 0.0)


