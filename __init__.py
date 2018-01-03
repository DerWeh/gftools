"""Collection of commonly used Green's functions and utilities.

So far mainly contains Bethe Green's functions.
Main purpose is to have a tested base.

FIXME: Add version via git!!!
"""
from __future__ import absolute_import, division, print_function

import numpy as np

from . import matrix


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
    """Local Green's function of Bethe lattice for infinite Coordination number.
    
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


def bethe_hilbert_transfrom(xi, half_bandwidth):
    r"""
    Hilbert transform of non-interacting DOS of the Bethe lattice.

    The Hilbert transform
    :math:`\tilde{D}(\xi) = \int_{-{}\infty}^{\infty}d\epsilon \frac{DOS(\epsilon)}{\xi - \epsilon}`
    takes for Bethe lattice in the limit of infinite coordination number the
    explicit form

    .. math::

        \tilde{D}(\xi) = 2*(\xi - s\sqrt{\xi^2 - D^2})/D^2

    with :math:`s=sgn[\Im{\xi}]`.

    see Georges_


    Params
    ------
    xi: array(complex), complex
        Point at which the Hilbert transform is evaluated
    half_bandwidth: double
        half-bandwidth of the DOS of the Bethe lattice

    .. _Georges: https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.68.13

    Notes
    -----
    Relation between nearest neighbor hopping t and half-bandwidth D:

    .. math::
        2t = D
    """
    xi_rel = xi/half_bandwidth
    # sgn = np.sign(xi_rel.imag)
    # return 2*(xi_rel - 1j*sgn*np.sqrt(1 - xi_rel**2))/half_bandwidth
    return 2.*xi_rel*(1 - np.sqrt(1 - xi_rel**-2))/half_bandwidth
