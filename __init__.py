"""Collection of commonly used Green's functions and utilities

So far mainly contains Bethe Green's functions.
Main purpose is to have a tested base.

FIXME: Add version via git!!!"""
from __future__ import absolute_import, division, print_function

import numpy as np


def bethe_dos(eps, half_bandwidth):
    """DOS of non-interacting Bethe lattice for infinite coordination number.
    
    Works by evaluating the complex function and truncating imaginary part.

    Params
    ------
    eps: array(double), double
        DOS is evaluated at points *eps*
    half_bandwidth: double
        half bandwidth of the DOS, DOS(|eps| > half_bandwidth) = 0
    """
    D = half_bandwidth
    eps = np.array(eps, dtype=np.complex256)
    res = np.sqrt(D**2 - eps**2) / (0.5 * np.pi * D**2)
    return res.real


def bethe_gf_omega(z, half_bandwidth):
    """Local Green's function of Bethe lattice for infinite Coordination number
    
    Params
    ------
    z: array(complex), complex
        Green's function is evaluated at complex frequency *z*
    half_bandwidth: double
        half-bandwidth of the DOS of the Bethe lattice
    """
    D = half_bandwidth
    return 2.*z*(1 - np.sqrt(1 - (D/z)**2))/D
    # return 2./z/(1 + np.sqrt(1 - (D**2)*(z**(-2))))
