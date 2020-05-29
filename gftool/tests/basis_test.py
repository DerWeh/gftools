"""Test functionality of basis representations."""
import numpy as np
import hypothesis.strategies as st
import pytest

from hypothesis import given, assume
from hypothesis_gufunc.gufunc import gufunc_args

from .context import gftool as gt


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@given(gufunc_args('(n)->(n)', dtype=np.complex_, elements=st.complex_numbers()))
def test_zp_to_ratpol(args):
    """Check that `gt.basis.ZeroPole` and `gt.basis.RatPol` give same result."""
    z, = args
    poles = np.array([1+1j, 0.2-3j])
    assume(not np.any(np.isclose(np.subtract.outer(z, poles), 0)))
    zeros = np.array([0, 1j])
    zp = gt.basis.ZeroPole(zeros, poles, amplitude=0.33+1.7j)
    ratpol = zp.to_ratpol()
    assert np.allclose(zp.eval(z), ratpol.eval(z), equal_nan=True)
