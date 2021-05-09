"""Test CPA."""
from functools import partial

import numpy as np
import hypothesis.strategies as st
import pytest

from hypothesis import given, assume
from hypothesis_gufunc.gufunc import gufunc_args

from .context import gftool as gt


@given(gufunc_args('(n)->(n)', dtype=np.complex_,
                   elements=st.complex_numbers(allow_infinity=False, allow_nan=False)))
def test_trival_cmpt_gf(args):
    """Test component Green's function for trivial case `concentration=1`."""
    z, = args
    assume(z.size > 0)
    assume(np.all(z.imag != 0))
    z = np.where(z.imag < 0, z.conj(), z)
    e_onsite = [-.53, .124]
    concentration = [1, 0]
    hilbert = partial(gt.bethe_gf_z, half_bandwidth=1)

    self_cpa_z = gt.cpa.solve_root(z, e_onsite, concentration, hilbert_trafo=hilbert,
                                   options=dict(fatol=1e-14))
    gf_cmpt_z = gt.cpa.gf_cmpt_z(z, self_cpa_z, e_onsite, hilbert_trafo=hilbert)[..., 0]
    assert np.allclose(gf_cmpt_z, hilbert(z - e_onsite[0]))
    assert np.allclose(e_onsite[0], self_cpa_z)


@given(z=st.complex_numbers(max_magnitude=1e-6))
def test_average(z):
    """Coherent Green's functions equals average of components at fixed-point."""
    assume(z.imag != 0)
    z = z.conjugate() if z.imag < 0 else z
    e_onsite = [-.53, .124]
    concentration = [0.34, 0.76]
    hilbert = partial(gt.bethe_gf_z, half_bandwidth=1)

    self_cpa_z = gt.cpa.solve_root(z, e_onsite, concentration, hilbert_trafo=hilbert)
    gf_cmpt_z = gt.cpa.gf_cmpt_z(z, self_cpa_z, e_onsite, hilbert_trafo=hilbert)
    gf_coher_z = hilbert(z - self_cpa_z)
    assert np.allclose(np.average(gf_cmpt_z, weights=concentration, axis=-1), gf_coher_z)


@pytest.mark.filterwarnings("ignore:Ill-conditioned matrix:scipy.linalg.LinAlgWarning")
@given(z=st.complex_numbers(max_magnitude=1e-6))
def test_restriction(z):
    """Check if restricted results yield physical results."""
    assume(z.imag != 0)
    z = z.conjugate() if z.imag < 0 else z
    e_onsite = [-.53, .124]
    concentration = [0.34, 0.76]
    hilbert = partial(gt.bethe_gf_z, half_bandwidth=1)

    self_cpa_z = gt.cpa.solve_root(z, e_onsite, concentration, hilbert_trafo=hilbert,
                                   options=dict(fatol=1e-14))
    assert self_cpa_z.imag < +1e-10


def test_cpa_occ():
    """Simple check if `gt.cpa.solve_fxdocc_root` gives right occupation."""
    beta = 7
    occ = 0.37
    iws = gt.matsubara_frequencies(range(256), beta=beta)
    hilbert = partial(gt.bethe_gf_z, half_bandwidth=1)
    e_onsite = [-.53, .124]
    concentration = [0.34, 0.76]
    self_iw, mu = gt.cpa.solve_fxdocc_root(iws, e_onsite, concentration,
                                           hilbert_trafo=hilbert, beta=beta, occ=occ,
                                           options=dict(fatol=1e-14))
    gf_coher_iw = hilbert(iws - self_iw)
    pot = np.average(e_onsite, weights=concentration) - mu
    occ_coher = gt.density_iw(iws, gf_coher_iw, moments=[1.0, pot], beta=beta)
    assert np.allclose(occ_coher, occ)
    gf_cmpt_iw = gt.cpa.gf_cmpt_z(iws, self_iw, e_onsite-mu, hilbert_trafo=hilbert).T
    occ_cmpt = gt.density_iw(iws, gf_cmpt_iw, moments=[1.0, pot], beta=beta)
    assert np.allclose(np.average(occ_cmpt, weights=concentration), occ)

    self_iw_fxdmu = gt.cpa.solve_root(iws, e_onsite-mu, concentration, hilbert_trafo=hilbert,
                                      options=dict(fatol=1e-14))
    assert np.allclose(self_iw_fxdmu, self_iw)
