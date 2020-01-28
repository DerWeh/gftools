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

from gftools._version import get_versions

__version__ = get_versions()['version']


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


bethe_dos = numba.vectorize(bethe_dos_)
fermi_fct = numba.vectorize(fermi_fct_)
fermi_fct_d1 = numba.vectorize(fermi_fct_d1_)
