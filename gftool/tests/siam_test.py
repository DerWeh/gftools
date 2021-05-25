"""Test for the single impurity Anderson model."""
import numpy as np
import hypothesis.strategies as st

from hypothesis import given, assume
from hypothesis_gufunc.gufunc import gufunc_args

from .context import gftool as gt


@given(gufunc_args('(),(n),(n)->(l)', dtype=np.float_,
                   elements=[st.floats(min_value=-1, max_value=1),
                             st.floats(min_value=-1, max_value=1),
                             st.floats(min_value=+0, max_value=1),
                             ],
                   max_dims_extra=2, max_side=5),
       st.complex_numbers(min_magnitude=1e-6, allow_infinity=False),
       )
def test_gf_loc_z_vs_resolvent(args, z):
    """Check `gf0_loc_z` against resolvent."""
    e_onsite, e_bath, hopping = args
    assume(abs(z.imag) > 1e-3)
    z = np.conj(z) if z.imag < 0 else z
    gf0_loc_z = gt.siam.gf0_loc_z(z, e_onsite, e_bath, abs(hopping)**2)
    ham = gt.siam.hamiltonian_matrix(e_onsite, e_bath, hopping)
    resolvent = np.linalg.inv(z*np.eye(*ham.shape[-2:]) - ham)
    assert np.allclose(gf0_loc_z, resolvent[..., 0, 0])


@given(gufunc_args('(),(n),(n)->(l)', dtype=np.float_,
                   elements=[st.floats(min_value=-1, max_value=1),
                             st.floats(min_value=-1, max_value=1),
                             st.floats(min_value=+0, max_value=1),
                             ],
                   max_dims_extra=2, max_side=5),)
def test_consistency_gf_loc_z_vs_ret_t(args):
    """Check if Laplace transform of `gf0_loc_ret_t` matches `gf0_loc_z`."""
    e_onsite, e_bath, hopping = args
    ww = np.linspace(-2, 2, num=51) + 1.0j
    tt = np.linspace(0, 25, 1001)  # eta=0.5 -> ~1e-10 at tt=50
    gf0_loc_z = gt.siam.gf0_loc_z(ww, e_onsite[..., None], e_bath[..., None, :],
                                  abs(hopping[..., None, :])**2)
    gf0_loc_ret_t = gt.siam.gf0_loc_ret_t(tt, e_onsite[..., None], e_bath[..., None, :],
                                          hopping[..., None, :])
    gf0_ft_z = gt.fourier.tt2z(tt, gf0_loc_ret_t, z=ww)
    assert np.allclose(gf0_loc_z, gf0_ft_z, rtol=1e-3, atol=1e-4)


@given(gufunc_args('(),(),(n),(n)->(l)', dtype=np.float_,
                   elements=[st.floats(min_value=+0, max_value=1000),
                             st.floats(min_value=-1, max_value=1),
                             st.floats(min_value=-1, max_value=1),
                             st.floats(min_value=+0, max_value=1),
                             ],
                   max_dims_extra=2, max_side=5),
       st.floats(min_value=0, exclude_min=True, allow_infinity=False))
def test_consistency_gf_loc_ret_vs_grle(args, beta):
    """Check if Laplace transform of `gf0_loc_ret_t` matches `gf0_loc_gr_t`/`gf0_loc_le_t`."""
    tt, e_onsite, e_bath, hopping = args
    gf0_loc_ret_t = gt.siam.gf0_loc_ret_t(tt, e_onsite, e_bath, hopping)
    gf0_loc_gr_t = gt.siam.gf0_loc_gr_t(tt, e_onsite, e_bath, hopping, beta=beta)
    gf0_loc_le_t = gt.siam.gf0_loc_le_t(tt, e_onsite, e_bath, hopping, beta=beta)
    assert np.allclose(gf0_loc_ret_t, gf0_loc_gr_t - gf0_loc_le_t)
