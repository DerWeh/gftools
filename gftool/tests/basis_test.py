"""Test functionality of basis representations."""
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given

import gftool as gt
from gftool.tests.custom_strategies import gufunc_args

assert_allclose = np.testing.assert_allclose


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@given(
    gufunc_args(
        shape_kwds={"signature": "(n)->(n)"},
        dtype=np.complex128,
        elements=st.complex_numbers(),
    )
)
def test_zp_to_ratpol(guargs):
    """Check that `gt.basis.ZeroPole` and `gt.basis.RatPol` give same result."""
    (z,) = guargs.args
    poles = np.array([1+1j, 0.2-3j])
    assume(not np.any(np.isclose(np.subtract.outer(z, poles), 0)))
    zeros = np.array([0, 1j])
    zp = gt.basis.ZeroPole(zeros, poles, amplitude=0.33+1.7j)
    ratpol = zp.to_ratpol()
    assert_allclose(zp.eval(z), ratpol.eval(z), equal_nan=True)
