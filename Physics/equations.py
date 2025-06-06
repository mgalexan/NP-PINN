import torch as t
import torch.func as F
import numpy as np

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



