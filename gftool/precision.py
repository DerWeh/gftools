"""Helper to temporarily increase working precision for calculations.

Note that in windows no quad-precision is available.

"""
import warnings

import numpy as np

try:
    np.complex256  # noqa: B018
    np.float128  # noqa: B018
except AttributeError:
    HAS_QUAD = False
    warnings.warn("No quad precision datatypes available!\n"
                  "Some functions might be less accurate.",
                  stacklevel=1)
    np.float128 = np.longdouble
    np.complex256 = np.clongdouble
else:
    HAS_QUAD = True

PRECISE_TYPES = {np.dtype(np.longdouble), np.dtype(np.clongdouble)}
