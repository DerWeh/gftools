"""Helper functions used throughout the code basis."""
import numpy as np


def _gu_sum(a, **kwds):
    """Sum over last axis for the use in generalized ufuncs.

    `np.sum` is more accurate over the fast axis,
    so we ensure the array to be c-contiguous.
    """
    return np.sum(np.ascontiguousarray(a), axis=-1, **kwds)
