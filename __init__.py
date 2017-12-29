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
