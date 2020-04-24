"""Reimplementation of `gftools` for the use with `numba`.

This is meant for performance crucial parts. It is less maintained then the
regular part of the module. Thus it is always recommended to us `gftools`,
only if performance has been proven to be insufficient, switch to `gftools.numba`

This module mirrors `gftools`, however not all functionality is implemented
(yet). The undecorated version of all functions exist, in case other decorations
are favorable.

"""
import numpy as np
import numba

import gftool as gt

__version__ = gt.__version__


def fermi_fct_(eps, beta):
    r"""Return the Fermi function :math:`1/(\exp(βz)+1)`.

    Parameters
    ----------
    eps : float
        The energy at which the Fermi function is evaluated.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.

    Returns
    -------
    fermi_fct : float
        The Fermi function.

    """
    # return 1./(np.exp(eps*beta) + 1.)
    return 0.5 * (1. + np.tanh(-0.5 * beta * eps))
    # return expit(-eps*beta)


def fermi_fct_d1_(eps, beta):
    r"""Return the 1st derivative of the Fermi function.

    .. math:: `-β\exp(βz)/{(\exp(βz)+1)}^2`

    Parameters
    ----------
    eps : float
        The energy at which the Fermi function is evaluated.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.

    Returns
    -------
    fermi_fct_d1 : float
        The derivative of the Fermi function.

    """
    fermi = fermi_fct(eps, beta)
    return -beta*fermi*(1-fermi)


def bethe_dos_(eps, half_bandwidth):
    """DOS of non-interacting Bethe lattice for infinite coordination number.

    Parameters
    ----------
    eps : float
        DOS is evaluated at points `eps`.
    half_bandwidth : float
        Half-bandwidth of the DOS, DOS(| `eps` | > `half_bandwidth`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`

    Returns
    -------
    result : float
        The value of the DOS.

    """
    D2 = half_bandwidth * half_bandwidth
    eps2 = eps*eps
    if eps2 < D2:
        return np.sqrt(D2 - eps2) / (0.5 * np.pi * D2)
    return 0.  # outside of bandwidth


def bethe_gf_omega_(z, half_bandwidth):
    """Local Green's function of Bethe lattice for infinite coordination number.

    Parameters
    ----------
    z : complex ndarray or complex
        Green's function is evaluated at complex frequency `z`
    half_bandwidth : float
        half-bandwidth of the DOS of the Bethe lattice
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`

    Returns
    -------
    bethe_gf_omega : complex ndarray or complex
        Value of the Green's function

    TODO: source

    """
    z_rel = np.array(z / half_bandwidth, dtype=numba.complex128)
    gf_z = 2./half_bandwidth*z_rel*(1 - np.sqrt(1 - z_rel**-2))
    return gf_z


def surface_gf_(z, eps, hopping_nn):
    r"""Surface Green's function for stacked layers.

    .. math::
        \left(1 - \sqrt{1 - 4 t^2 g_{00}^2}\right)/(2 t^2 g_{00})

    with :math:`g_{00} = (z-ϵ)^{-1}` [6]_. This is in principle the Green's function
    for a semi-infinite chain.

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
    surface_gf : complex
        Value of the surface Green's function

    References
    ----------
    .. [6] Odashima, Mariana M., Beatriz G. Prado, and E. Vernek. Pedagogical
    Introduction to Equilibrium Green's Functions: Condensed-Matter Examples
    with Numerical Implementations. Revista Brasileira de Ensino de Fisica 39,
    no. 1 (September 22, 2016).
    https://doi.org/10.1590/1806-9126-rbef-2016-0087.

    """
    return bethe_gf_omega(z-eps, 2.*hopping_nn)


def surface_dos_(eps, hopping_nn):
    r"""Surface DOS for non-interacting stacked layers.

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
    surface_dos : float
        Value of the surface Green's function

    """
    return bethe_dos(eps, 2.*hopping_nn)


bethe_dos = numba.vectorize(nopython=True)(bethe_dos_)
fermi_fct = numba.vectorize(fermi_fct_)
fermi_fct_d1 = numba.vectorize(fermi_fct_d1_)
bethe_gf_omega = numba.vectorize(nopython=True)(bethe_gf_omega_)
surface_gf = numba.vectorize(nopython=True)(surface_gf_)
surface_dos = numba.vectorize(nopython=True)(surface_dos_)
