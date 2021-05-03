"""Helper functions used throughout the code basis."""
from functools import partial

import numpy as np

from mpmath import fp


_ellipk_z = np.frompyfunc(partial(fp.ellipf, np.pi/2), 1, 1)


def _gu_sum(a, **kwds):
    """Sum over last axis for the use in generalized ufuncs.

    `np.sum` is more accurate over the fast axis,
    so we ensure the array to be c-contiguous.
    """
    return np.sum(np.ascontiguousarray(a), axis=-1, **kwds)


def _gu_matvec(x1, x2):
    """Matrix-vector product for the use in generalized ufuncs.

    Parameters
    ----------
    x1 : (..., N, M) np.ndarray
        The matrix.
    x2 : (..., M) np.ndarray
        The vector.

    Returns
    -------
    (..., N) np.ndarray
        The resulting vector.

    """
    return (x1 @ x2[..., np.newaxis])[..., 0]


def _u_ellipk(z):
    """Complete elliptic integral of first kind `scipy.special.ellip` for complex arguments.

    Wraps the `mpmath` implementation `mpmath.fp.ellipf` using `numpy.frompyfunc`.

    Parameters
    ----------
    z : complex or complex array_like
        Complex argument

    Returns
    -------
    complex np.ndarray or complex
        The complete elliptic integral.

    """
    ellipk = _ellipk_z(np.asarray(z, dtype=complex))
    try:
        ellipk = ellipk.astype(complex)
    except AttributeError:  # complex not np.ndarray
        pass
    return ellipk
