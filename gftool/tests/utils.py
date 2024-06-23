"""Utilities reused in tests."""

import hypothesis.strategies as st
import numpy as np


@st.composite
def st_complex(draw, *, min_magnitude=0, max_magnitude=None, allow_infinity=None, allow_nan=None):
    """Return a strategy which generates complex numbers."""
    real = draw(st.floats(allow_nan=allow_nan, allow_infinity=allow_infinity))
    imag = draw(st.floats(allow_nan=allow_nan, allow_infinity=allow_infinity))
    # proper shrinkage: throw always max_magnitude < value and value < min_magnitude
    cmp = real + 1j*imag
    mag = abs(cmp)
    if min_magnitude < 0:
        raise ValueError
    # not that this is not
    if 0 < mag < min_magnitude:
        # increase magnitude to min
        return cmp * (mag + min_magnitude) / mag
    if max_magnitude is not None and max_magnitude < mag:
        # reduce magnitude to max
        return cmp * max_magnitude / (mag + max_magnitude)

    return cmp


def assert_allclose_vm(actual, desired, rtol=1e-7, atol=1e-14, **kwds):
    """Relax `assert_allclose` in case some elements are huge and others 0."""
    # TODO: we should provide an axis argument and somehow iterate...
    # for now we just relax the test
    fact = np.maximum(np.linalg.norm(desired), 1.0)
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol*fact, **kwds)
