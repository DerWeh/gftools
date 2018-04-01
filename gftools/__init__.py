"""Collection of commonly used Green's functions and utilities.

So far mainly contains Bethe Green's functions.
Main purpose is to have a tested base.

Subpackages
-----------
    matrix   --- Work with Green's functions in matrix form, mainly for r-DMFT

.. _Georges et al:
    https://doi.org/10.1103/RevModPhys.68.13
"""
from __future__ import absolute_import, division, print_function

import numpy as np

from ._version import get_versions

__version__ = get_versions()['version']


def bethe_dos(eps, half_bandwidth):
    """DOS of non-interacting Bethe lattice for infinite coordination number.
    
    Parameters
    ----------
    eps : array(double), double
        DOS is evaluated at points `eps`.
    half_bandwidth : double
        Half bandwidth of the DOS, DOS(| `eps` | > `half_bandwidth`) = 0.

    Returns
    -------
    bethe_dos : array(double), double
        The value of the DOS.

    """
    D2 = half_bandwidth * half_bandwidth
    eps2 = eps*eps
    mask = eps2 < D2
    try:
        result = np.empty_like(eps)
        result[~mask] = 0
    except IndexError:  # eps is scalar
        if mask:
            return np.sqrt(D2 - eps2) / (0.5 * np.pi * D2)
        return 0.  # outside of bandwidth
    else:
        result[mask] = np.sqrt(D2 - eps2[mask]) / (0.5 * np.pi * D2)
        return result


def bethe_gf_omega(z, half_bandwidth):
    """Local Green's function of Bethe lattice for infinite Coordination number.
    
    Parameters
    ----------
    z: array(complex), complex
        Green's function is evaluated at complex frequency `z`
    half_bandwidth: double
        half-bandwidth of the DOS of the Bethe lattice

    Returns
    -------
    bethe_gf_omega : array(complex), complex
        Value of the Green's function

    """
    z_rel = z / half_bandwidth
    return 2./half_bandwidth*z_rel*(1 - np.sqrt(1 - 1/(z_rel*z_rel)))


def bethe_hilbert_transfrom(xi, half_bandwidth):
    r"""Hilbert transform of non-interacting DOS of the Bethe lattice.

    FIXME: the lattice Hilbert transform is the same as the non-interacting
        Green's function.

    The Hilbert transform

    .. math::
        \tilde{D}(\xi) = \int_{-\infty}^{\infty}d\epsilon \frac{DOS(\epsilon)}{\xi - \epsilon}

    takes for Bethe lattice in the limit of infinite coordination number the
    explicit form

    .. math::
        \tilde{D}(\xi) = 2*(\xi - s\sqrt{\xi^2 - D^2})/D^2

    with :math:`s=sgn[\Im{\xi}]`.
    See `Georges et al`_.


    Parameters
    ----------
    xi : array(complex), complex
        Point at which the Hilbert transform is evaluated
    half_bandwidth : double
        half-bandwidth of the DOS of the Bethe lattice

    Returns
    -------
    bethe_hilbert_transfrom : array(complex), complex
        Hilbert transform of `xi`.

    Notes
    -----
    Relation between nearest neighbor hopping `t` and half-bandwidth `D`

    .. math::
        2t = D

    """
    return bethe_gf_omega(xi, half_bandwidth)
