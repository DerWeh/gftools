"""Test the utility functions related to Green's functions."""
from functools import lru_cache, partial
from warnings import catch_warnings, filterwarnings

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given
from mpmath import fp

import gftool as gt
from gftool.basis import pole
from gftool.tests.custom_strategies import gufunc_args

approx = partial(np.allclose, rtol=1e-12, atol=1e-16, equal_nan=True)
assert_allclose = np.testing.assert_allclose
# cache pade_frequencies
gt.statistics._pade_frequencies = lru_cache(maxsize=10)(gt.statistics._pade_frequencies)


def test_bose_edge_cases():
    """Check exact limits at `0` and `np.infty`."""
    assert gt.bose_fct(0., 1) == np.inf
    assert gt.bose_fct(np.inf, 1) == 0


@given(z=st.floats(min_value=1e-4, max_value=1e4), n=st.integers(min_value=-100, max_value=100))
@pytest.mark.parametrize("beta", [0.7, 1.38, 1000])
def test_complex_bose(z, n, beta):
    """
    Check if Bose function has imaginary periodicity.

    For bosonic Matsubara frequencies :math:`f(z+iν_n) = f(z)` should hold.
    """
    iv_n = gt.matsubara_frequencies_b(n, beta=beta)
    bose_cmpx = gt.bose_fct(z+iv_n, beta=beta)
    bose_real = gt.bose_fct(z, beta=beta)
    assert_allclose(bose_cmpx.real, bose_real)
    assert bose_cmpx.imag < 1e-6*max(1, bose_real)


@given(z=st.floats())
@pytest.mark.parametrize("beta", [0.7, 1.38, 1000])
def test_fermi(z, beta):
    """Check if Fermi functions agrees with the standard form."""
    with catch_warnings():  # fermi_fct should avoid these overflows
        filterwarnings("ignore", message="overflow encountered in exp", category=RuntimeWarning)
        fermi_comp = 1./(np.exp(beta*z) + 1)
    # assert approx(fermi_comp) == gt.fermi_fct(z, beta=beta)
    assert approx(gt.fermi_fct(z, beta=beta), fermi_comp)


@pytest.mark.filterwarnings("ignore:(overflow)|(invalid value):RuntimeWarning")
@given(z=st.floats(), n=st.integers(min_value=-100, max_value=100))
@pytest.mark.parametrize("beta", [0.7, 1.38, 1000])
def test_complex_fermi(z, n, beta):
    """
    Check if Fermi function has imaginary periodicity.

    For bosonic Matsubara frequencies :math:`f(z+iν_n) = f(z)` should hold.
    """
    iv_n = gt.matsubara_frequencies_b(n, beta=beta)
    fermi_cmpx = gt.fermi_fct(z+iv_n, beta=beta)
    fermi_real = gt.fermi_fct(z, beta=beta)
    assert approx(fermi_cmpx.real, fermi_real)
    if not np.isnan(fermi_cmpx.real):
        assert fermi_cmpx.imag < 1e-6


@given(z=st.floats(min_value=0., max_value=1.))
@pytest.mark.parametrize("beta", [0.7, 1.38, 1000])
def test_inverse_fermi(z, beta):
    """
    Check that `gt.fermi_fct_inv` is indeed the inverse.

    The other direction does not work in general, due to rounding.
    """
    assert approx(gt.fermi_fct(gt.fermi_fct_inv(z, beta), beta), z)


@given(z=st.floats())
@pytest.mark.parametrize("beta", [0.7, 1.38, 1000])
def test_fermi_d1_std_form(z, beta):
    """Check if Fermi functions agrees with the standard form."""
    assume(z*beta < 500.)  # avoid overflows in naive implementation
    exp = np.exp(beta*z)
    with catch_warnings():  # fermi_fct_d1 should avoid these overflows
        filterwarnings("ignore", message="overflow encountered in", category=RuntimeWarning)
        fermi_d1_comp = -beta*exp/(exp+1)**2
    assert approx(gt.fermi_fct_d1(z, beta=beta), fermi_d1_comp)


@given(z=st.complex_numbers(allow_infinity=False, max_magnitude=1e2))  # quad doesn't handle inf
def test_fermi_derivative_1(z):
    """Check if integrated derivative yields the original function."""
    assume(abs(z.real) > 1e-4)  # avoid imaginary axis
    # make sure to be away from poles
    if abs(z.real) < 0.5:
        zimag_per = (z.imag - np.pi) % (2*np.pi)
        dist = min(zimag_per, 2*np.pi - zimag_per)
    else:
        dist = abs(z.real) / 2
    assert_allclose(fp.diff(partial(gt.fermi_fct, beta=1), z, method="quad", radius=dist/2),
                    gt.fermi_fct_d1(z, beta=1), atol=1e-12)


@given(eps=st.floats(min_value=-1e4, max_value=1e4),
       occ=st.floats(min_value=1e-4, max_value=1 - 1e-4))
def test_chemical_potential_single_pole(eps, occ):
    """Test chemical potential search for single pole Green's function."""
    BETA = 25

    # FIXME: Fermi function fails for:
    #        test_chemical_potential_single_pole(eps=0.0, occ=5e-324)
    #        test_chemical_potential_single_pole(eps=65535.0, occ=1e-06)
    #        test_chemical_potential_single_pole(eps=1.2676506002282296e+24, occ=1e-06)

    def occ_fct(mu):
        return gt.fermi_fct(eps-mu, beta=BETA)

    mu = gt.chemical_potential(lambda mu: occ_fct(mu) - occ)
    assert_allclose(occ_fct(mu), occ)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # test gt.density while it's still there
def test_density():
    """Check density for simple Green's functions."""
    beta = 10.1
    D = 1.2
    iws = gt.matsubara_frequencies(np.arange(2**14), beta=beta)

    #
    # Bethe lattice Gf
    #
    bethe_half = gt.bethe_gf_z(iws, half_bandwidth=D)
    assert gt.density(bethe_half, potential=0, beta=beta)[0] == 0.5
    assert gt.density_iw(iws, bethe_half, moments=[1.], beta=beta) == 0.5

    large_shift = 10000*D
    # Bethe lattice almost filled (depends on small temperature)
    bethe_full = gt.bethe_gf_z(iws+large_shift, half_bandwidth=D)
    assert gt.density(bethe_full, potential=large_shift, beta=beta)[0] == pytest.approx(1.0)
    assert_allclose(gt.density_iw(iws, bethe_full, moments=[1., -large_shift], beta=beta), 1.)
    # Bethe lattice almost empty (depends on small temperature)
    bethe_empty = gt.bethe_gf_z(iws-large_shift, half_bandwidth=D)
    assert gt.density(bethe_empty, potential=-large_shift, beta=beta)[0] \
        == pytest.approx(0.0, abs=1e-6)
    assert_allclose(gt.density_iw(iws, bethe_empty, moments=[1., +large_shift], beta=beta),
                    0., atol=1e-8)

    #
    # single site
    #
    assert gt.density(1./iws, potential=0, beta=beta)[0] == pytest.approx(0.5)
    assert_allclose(gt.density_iw(iws, 1./iws, moments=[1., 0], beta=beta), 0.5)


@given(
    guargs=gufunc_args(
        # we do not test for broadcasting here (max_dims=0)
        shape_kwds={"signature": "(n),(n)->(n)", "max_dims": 0},
        dtype=np.float64,
        elements=[
            st.floats(min_value=-10, max_value=10),
            st.floats(min_value=0, max_value=10),
        ],
    )
)
def test_density_iw(guargs):
    """Check `gt.density_iw` on Matsubara frequencies for multi pole Green's function."""
    beta = 17
    poles, residues = guargs.args
    iw = gt.matsubara_frequencies(range(4096), beta=beta)
    if np.any(residues.sum(axis=-1) > 10.):
        # there are issues with moments with large residues, without moments it's fine
        residues /= residues.sum(axis=-1, keepdims=True)
    gf_poles = pole.PoleGf(poles=poles, residues=residues)
    gf_iw = gf_poles.eval_z(iw)
    moments = gf_poles.moments([1, 2, 3, 4])
    occ_ref = gf_poles.occ(beta)
    occ = gt.density_iw(iw, gf_iw, beta=beta, moments=moments)
    assert_allclose(occ, occ_ref, atol=1e-5)
    # add moment
    occ = gt.density_iw(iw, gf_iw, beta=beta, moments=moments, n_fit=4)
    assert_allclose(occ, occ_ref, atol=1e-6)


@pytest.fixture(scope="module")
def pade_frequencies():
    """Provide Padé frequency as they are slow to calculate."""
    izp, rp = gt.pade_frequencies(100, beta=1)
    izp.flags.writeable = False
    rp.flags.writeable = False

    def pade_frequencies_(beta):
        return izp / beta, rp

    return pade_frequencies_


@given(
    guargs=gufunc_args(
        shape_kwds={"signature": "(n),(n)->(n)"},
        dtype=np.float64,
        elements=[
            st.floats(min_value=-10, max_value=10),
            st.floats(min_value=0, max_value=10),
        ],
    ),
)
def test_density_izp(guargs, pade_frequencies):
    """Check `gt.density_iw` on Padé frequencies for multi pole Green's function."""
    beta = 17
    poles, residues = guargs.args
    izp, rp = pade_frequencies(beta)
    if np.any(residues.sum(axis=-1) > 10.):
        # there are issues with moments with large residues, without moments it's fine
        residues /= residues.sum(axis=-1, keepdims=True)
    gf_izp = gt.pole_gf_z(izp, poles[..., np.newaxis, :], residues[..., np.newaxis, :])
    gf_poles = pole.PoleGf(poles=poles, residues=residues)
    occ_ref = gf_poles.occ(beta)
    occ = gt.density_iw(izp, gf_izp, weights=rp, beta=beta,
                        moments=residues.sum(axis=-1, keepdims=True))
    assert_allclose(occ, occ_ref, atol=1e-12)
    # add moment
    moments = gf_poles.moments([1, 2, 3])
    occ = gt.density_iw(izp, gf_izp, weights=rp, beta=beta,
                        moments=moments)
    assert_allclose(occ, occ_ref, atol=1e-5)
