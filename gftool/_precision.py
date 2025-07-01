"""
Helper to temporarily increase working precision for calculations.

Note that in windows no quad-precision is available.
"""

import warnings

import numpy as np

try:
    float128 = np.float128
    complex256 = np.complex256
except AttributeError:
    HAS_QUAD = False
    warnings.warn(
        "No quad precision data types available!\n"
        "Some functions might be less accurate.",
        stacklevel=1,
    )
    float128 = np.longdouble
    complex256 = np.clongdouble
else:
    HAS_QUAD = True

PRECISE_TYPES = {np.dtype(np.longdouble), np.dtype(np.clongdouble)}
