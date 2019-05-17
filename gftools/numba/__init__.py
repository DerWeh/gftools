"""Reimplementation of `gftools` for the use with `numba`.

This is meant for performance crucial parts. It is less maintained then the
regular part of the module. Thus it is always recommended to us `gftools`,
only if performance as been proven to be insufficient, switch to `gftools.numba`

This module mirrors `gftools`, however not all functionality is implemented
(yet). The undecorated version of all functions exist, in case other decorations
are favorable.

"""
import numpy as np
import numba

from gftools._version import get_versions

__version__ = get_versions()['version']


def bethe_dos_(eps, half_bandwidth):
    """DOS of non-interacting Bethe lattice for infinite coordination number.

    Parameters
    ----------
    eps : float ndarray or float
        DOS is evaluated at points `eps`.
    half_bandwidth : float
        Half-bandwidth of the DOS, DOS(| `eps` | > `half_bandwidth`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`

    Returns
    -------
    result : float ndarray or float
        The value of the DOS.

    """
    D2 = half_bandwidth * half_bandwidth
    eps2 = eps*eps
    if eps2 < D2:
        return np.sqrt(D2 - eps2) / (0.5 * np.pi * D2)
    return 0.  # outside of bandwidth


bethe_dos = numba.vectorize(bethe_dos_)
