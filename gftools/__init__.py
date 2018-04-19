# encoding: utf-8
"""Collection of commonly used Green's functions and utilities.

So far mainly contains Bethe Green's functions.
Main purpose is to have a tested base.

Subpackages
-----------
    matrix   --- Work with Green's functions in matrix form, mainly for r-DMFT

.. _Georges et al:
    https://doi.org/10.1103/RevModPhys.68.13
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math

import numpy as np

from ._version import get_versions

__version__ = get_versions()['version']


def matsubara_frequencies(n_points, beta):
    r"""Return *fermionic* Matsubara frequencies :math:`iω_n` for the points `n_points`.

    Parameters
    ----------
    n_poins : array(int)
        Points for which the Matsubara frequencies :math:`iω_n` are returned.
    beta : float
        Inverse temperature `beta` = 1/T

    Returns
    -------
    matsubara_frequencies : array(complex)
        Array of the imaginary Matsubara frequencies

    """
    return 1j * np.pi * (2*n_points + 1) / beta


def bethe_dos(eps, half_bandwidth):
    """DOS of non-interacting Bethe lattice for infinite coordination number.
    
    Parameters
    ----------
    eps : array(float), float
        DOS is evaluated at points `eps`.
    half_bandwidth : float
        Half-bandwidth of the DOS, DOS(| `eps` | > `half_bandwidth`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`

    Returns
    -------
    bethe_dos : array(float), float
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
    z : array(complex), complex
        Green's function is evaluated at complex frequency `z`
    half_bandwidth : float
        half-bandwidth of the DOS of the Bethe lattice
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`

    Returns
    -------
    bethe_gf_omega : array(complex), complex
        Value of the Green's function

    TODO: source

    """
    z_rel = z / half_bandwidth
    return 2./half_bandwidth*z_rel*(1 - np.sqrt(1 - 1/(z_rel*z_rel)))


def bethe_hilbert_transfrom(xi, half_bandwidth):
    r"""Hilbert transform of non-interacting DOS of the Bethe lattice.

    FIXME: the lattice Hilbert transform is the same as the non-interacting
        Green's function.

    The Hilbert transform

    .. math::
        \tilde{D}(ξ) = ∫_{-∞}^{∞}dϵ \frac{DOS(ϵ)}{ξ − ϵ}

    takes for Bethe lattice in the limit of infinite coordination number the
    explicit form

    .. math::
        \tilde{D}(ξ) = 2*(ξ - s\sqrt{ξ^2 - D^2})/D^2

    with :math:`s=sgn[ℑ{ξ}]`.
    See `Georges et al`_.


    Parameters
    ----------
    xi : array(complex), complex
        Point at which the Hilbert transform is evaluated
    half_bandwidth : float
        half-bandwidth of the DOS of the Bethe lattice

    Returns
    -------
    bethe_hilbert_transfrom : array(complex), complex
        Hilbert transform of `xi`.

    Note
    ----
    Relation between nearest neighbor hopping `t` and half-bandwidth `D`

    .. math::
        2t = D

    """
    return bethe_gf_omega(xi, half_bandwidth)


def bethe_surface_gf(z, eps, hopping_nn):
    r"""Surface Green's function for stacked layers of Bethe lattices.

    .. math::
        \left(1 - \sqrt{1 - 4 t^2 g_{00}^2}\right)/(2 t^2 g_{00})

    with :math:`g_{00} = (z-ϵ)^{-1}`

    TODO: source

    Parameters
    ----------
    z : complex
        Green's function is evaluated at complex frequency `z`.
    eps : float
        Eigenenergy (dispersion) for which the Green's function is evaluated.
    hopping_nn : float
        Nearest neighbor hopping `t` between neighboring layers.

    Returns
    -------
    bethe_surface_gf : complex
        Value of the surface Green's function

    """
    return bethe_gf_omega(z-eps, half_bandwidth=2.*hopping_nn)


def density(gf_iw, potential, sigma_max, beta):
    r"""Calculate the number density of the Green's function `gf_iw` at finite temperature `beta`.

    Parameters
    ----------
    gf_iw : array(complex)
        The Matsubara frequency Green's function for positive frequencies :math:`iω_n`.
    potential : float
        Is the on-site energy of the Green's function, which is a real constant
        shift of the frequencies :math:`iω_n`.
    sigma_max : complex
        The maximal calculated self-energy. `sigma_max` is assumed to be the constant
        part of the self-energy, the rest decays like :math:`1/ω`.
    beta : float
        The inverse temperature `beta` = 1/T.

    Returns
    -------
    density : float
        The number density of the given Green's function `gf_iw`.

    Notes
    -----
    The number density can be obtained from the Matsubara frequency Green's function using

    .. math:: n = 1/β ∑_{n=-∞}^{∞} G(iω_n)

    Green's functions, however, only decay like :math:`O(\omega)`,
    thus truncation of the summation yields a non-vanishing contribution of the tail.
    To take this into consideration the tail has to be summed analytically.
    The most general and leading contribution is :math:`1/(i\omega_n)`.
    The summation yields an additional contribution :math:`1/2`.

    References
    ----------
    .. [1] Hale, S. T. F., and J. K. Freericks. "Many-Body Effects on the
       Capacitance of Multilayers Made from Strongly Correlated Materials."
       Physical Review B 85, no. 20 (May 24, 2012).
       https://doi.org/10.1103/PhysRevB.85.205444.

    """
    raise NotImplementedError
    # iw = matsubara(beta, fermion, n=len(gf_iw))
    iw = None  # FIXME
    tail = iw - potential - sigma_max
    n = np.sum(gf_iw - 1/tail) / beta
    n += .5 + 0.5*math.tanh(0.5 * beta * (potential - sigma_max))
    return n
