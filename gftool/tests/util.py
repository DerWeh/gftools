"""Utilities reused in tests."""
import hypothesis.strategies as st


@st.composite
def st_complex(draw, *, min_magnitude=0, max_magnitude=None, allow_infinity=None, allow_nan=None):
    real = draw(st.floats(allow_nan=allow_nan, allow_infinity=allow_infinity))
    imag = draw(st.floats(allow_nan=allow_nan, allow_infinity=allow_infinity))
    # proper shrinkage: throw aways max_magnitude < value and value < min_magnitude
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
