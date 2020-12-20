"""Test for the single impurity Anderson model."""
import numpy as np
import pytest
import hypothesis.strategies as st

from hypothesis import given, assume
from hypothesis_gufunc.gufunc import gufunc_args

from .context import gftool as gt


@given(gufunc_args('(),(n),(n)->(l)', dtype=np.float_,
                   elements=[st.floats(min_value=-1, max_value=1),
                             st.floats(min_value=-1, max_value=1),
                             st.floats(min_value=+0, max_value=1),],
                   max_dims_extra=2, max_side=5),)
def test_consistency_gf_loc_z_vs_ret_t(args):
    e_onsite, e_bath, hopping = args
    ww = np.linspace(-2, 2, num=51) + 1.0j
    tt = np.linspace(0, 25, 1001)  # eta=0.5 -> ~1e-10 at tt=50
    gf0_loc_z = gt.siam.gf0_loc_z(ww, e_onsite[..., None], e_bath[..., None, :],
                                  abs(hopping[..., None, :])**2)
    gf0_loc_ret_t = gt.siam.gf0_loc_ret_t(tt, e_onsite[..., None], e_bath[..., None, :],
                                          hopping[..., None, :])
    gf0_ft_z = gt.fourier.tt2z(tt, gf0_loc_ret_t, z=ww)
    assert np.allclose(gf0_loc_z, gf0_ft_z, rtol=1e-3, atol=1e-4)
