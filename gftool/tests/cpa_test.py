"""Test CPA."""
from functools import partial
from itertools import product

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given

import gftool as gt
from gftool._precision import HAS_QUAD
from gftool.tests.custom_strategies import gufunc_args

assert_allclose = np.testing.assert_allclose

ignore_close_to_root = pytest.mark.filterwarnings(
    "ignore:(invalid value encountered in (double_scalars|scalar divide)):RuntimeWarning"
)
ignore_illconditioned = pytest.mark.filterwarnings(
    "ignore:(Ill-conditioned matrix):scipy.linalg.LinAlgWarning"
)


@given(
    gufunc_args(
        shape_kwds={"signature": "(n)->(n)"},
        dtype=np.complex128,
        elements=st.complex_numbers(
            allow_infinity=False,
            allow_nan=False,
            max_magnitude=None if HAS_QUAD else 1e100,
        ),
    )
)
def test_trival_cmpt_gf(guargs):
    """Test component Green's function for trivial case `concentration=1`."""
    (z,) = guargs.args
    assume(z.size > 0)
    assume(np.all(z.imag != 0))
    z = np.where(z.imag < 0, z.conj(), z)
    e_onsite = [-.53, .124]
    concentration = [1, 0]
    hilbert = partial(gt.bethe_gf_z, half_bandwidth=1)

    self_cpa_z = gt.cpa.solve_root(z, e_onsite, concentration, hilbert_trafo=hilbert,
                                   options={"fatol": 1e-14})
    gf_cmpt_z = gt.cpa.gf_cmpt_z(z, self_cpa_z, e_onsite, hilbert_trafo=hilbert)[..., 0]
    assert_allclose(gf_cmpt_z, hilbert(z - e_onsite[0]))
    assert_allclose(e_onsite[0], self_cpa_z)


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
    assert_allclose(np.average(gf_cmpt_z, weights=concentration, axis=-1), gf_coher_z, rtol=1e-5)


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
                                   options={"fatol": 1e-14})
    assert self_cpa_z.imag < +1e-10


@ignore_close_to_root
@given(
    gufunc_args(
        shape_kwds={"signature": "(),(n),(n)->()", "min_side": 2, "max_side": 4},
        dtype=complex,
        elements=[
            st.complex_numbers(max_magnitude=1e-3),
            st.floats(min_value=-2, max_value=+2),
            st.floats(min_value=0, max_value=1),
        ],
    )
)
def test_cpa_interface(guargs):
    """Test fixed-occupation CPA for various broadcastable inputs."""
    hilbert = partial(gt.bethe_gf_z, half_bandwidth=1)
    z, eps, conc = guargs.args
    assume(np.all(conc.sum(axis=-1) > 1e-12))
    z = np.where(z.imag < 0, z.conj(), z)
    z += 1j  # go into imaginary plane where convergences is fast
    conc = np.where(conc < np.finfo(conc.dtype).eps*2, 0, conc)
    conc /= conc.sum(axis=-1)[..., np.newaxis]
    # for some reason tiny values crashes root search...
    eps = np.where(abs(eps) < np.finfo(eps.dtype).eps*2, 0, eps)

    self_cpa_z = gt.cpa.solve_root(z, e_onsite=eps, concentration=conc, hilbert_trafo=hilbert)
    gf_z = hilbert(z - self_cpa_z)
    gf_cmpt_z = gt.cpa.gf_cmpt_z(z, self_cpa_z, eps, hilbert_trafo=hilbert)
    assert_allclose(np.sum(conc*gf_cmpt_z, axis=-1), gf_z, rtol=1e-3)


@ignore_illconditioned
@pytest.mark.parametrize("z", [np.linspace(-5, 5, num=251) + 1e-2j,
                               gt.matsubara_frequencies(range(256), beta=13)])
def test_cpa(z):
    """Test fixed-occupation CPA for a grid of parameters and odd starting points."""
    hilbert = partial(gt.bethe_gf_z, half_bandwidth=1)
    for conc, delta, mu in product([(0.5, 0.5), (0.3, 0.7), (0.1, 0.9)],  # conentration
                                   [0.4, 0.8, 1.2, 1.6],  # disorder strength
                                   [-1.5, -0.5, 0.5, 1.5]):  # chemical potential/filling
        eps = 0.5*np.array([-delta, +delta]) - mu
        self_cpa_z = gt.cpa.solve_root(z, eps, conc, hilbert_trafo=hilbert)
        gf_z = hilbert(z - self_cpa_z)
        gf_cmpt_z = gt.cpa.gf_cmpt_z(z, self_cpa_z, eps, hilbert_trafo=hilbert)
        assert_allclose(np.sum(conc*gf_cmpt_z, axis=-1), gf_z, rtol=1e-4)


def test_cpa_occ_simple():
    """Simple check if `gt.cpa.solve_fxdocc_root` gives right occupation."""
    beta = 7
    occ = 0.37
    iws = gt.matsubara_frequencies(range(256), beta=beta)
    hilbert = partial(gt.bethe_gf_z, half_bandwidth=1)
    e_onsite = [-.53, .124]
    concentration = [0.34, 0.76]
    self_iw, mu = gt.cpa.solve_fxdocc_root(iws, e_onsite, concentration,
                                           hilbert_trafo=hilbert, beta=beta, occ=occ,
                                           options={"fatol": 1e-14})
    gf_coher_iw = hilbert(iws - self_iw)
    pot = np.average(e_onsite, weights=concentration) - mu
    occ_coher = gt.density_iw(iws, gf_coher_iw, moments=[1.0, pot], beta=beta)
    assert_allclose(occ_coher, occ)
    gf_cmpt_iw = gt.cpa.gf_cmpt_z(iws, self_iw, np.array(e_onsite)-mu, hilbert_trafo=hilbert).T
    occ_cmpt = gt.density_iw(iws, gf_cmpt_iw, moments=[1.0, pot], beta=beta)
    assert_allclose(np.average(occ_cmpt, weights=concentration), occ)

    self_iw_fxdmu = gt.cpa.solve_root(iws, np.array(e_onsite)-mu, concentration,
                                      hilbert_trafo=hilbert, options={"fatol": 1e-14})
    assert_allclose(self_iw_fxdmu, self_iw)


@pytest.mark.parametrize("conc", [(0.5, 0.5), (0.3, 0.7), (0.1, 0.9)])
def test_cpa_occ(conc):
    """Test fixed-occupation CPA for a grid of parameters and odd starting points."""
    beta = 13.5
    izp, rp = gt.pade_frequencies(50, beta)
    hilbert = partial(gt.bethe_gf_z, half_bandwidth=1)
    conc = np.array(conc)
    for occ, delta, mu0 in product([0.2, 0.4, 0.6, 0.8],  # occupation
                                   [0.4, 0.8, 1.2, 1.6],  # disorder strength
                                   [-1.5, -0.5, 0.5, 1.5]):  # chemical potential
        eps = 0.5*np.array([-delta, +delta]) - mu0
        self_izp, mu = gt.cpa.solve_fxdocc_root(
            izp, eps[..., np.newaxis, :], conc, hilbert_trafo=hilbert, weights=rp,
            beta=beta, occ=occ,
        )
        gf_izp = hilbert(izp - self_izp)
        pot = np.sum(eps * conc, axis=-1) - mu
        mom = np.stack([np.ones_like(pot), pot], axis=-1)
        occ_coher = gt.density_iw(izp, gf_izp, weights=rp, moments=mom, beta=beta).sum()
        assert occ == pytest.approx(occ_coher, abs=1e-6)
