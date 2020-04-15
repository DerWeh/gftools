"""Helper to temporarily increase working precision for calculations.

Note that in windows no quad-precision is available.

"""
import warnings

import numpy as np

try:
    # pylint: disable=pointless-statement
    np.complex256
    np.float128
except AttributeError:
    HAS_QUAD = False
    warnings.warn("No quad precision datatypes available!\n"
                  "Some functions might be less accurate.")
    np.float128 = np.longfloat
    np.complex256 = np.clongfloat
else:
    HAS_QUAD = True

PRECISE_TYPES = {np.dtype(np.longfloat), np.dtype(np.clongfloat)}
