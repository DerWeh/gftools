"""Helper functions used throughout the code basis."""
import numpy as np


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
