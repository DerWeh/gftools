"""Test the Fourier transformation of Green's functions.

TODO: This module urgently needs a refactor for `tt2z`. We have too many
different methods.
"""
# pylint: disable=protected-access
from functools import partial

import numpy as np
import pytest
import hypothesis.strategies as st

from hypothesis import given, assume
from hypothesis.extra.numpy import arrays
from hypothesis_gufunc.gufunc import gufunc_args
from scipy.integrate import simpson

from .context import gftool as gt

assert_allclose = np.testing.assert_allclose

assert_allclose = np.testing.assert_allclose


@given(gufunc_args('(n)->(n),(m)', dtype=np.float_,
                   elements=st.floats(min_value=-1e6, max_value=1e6),
                   max_dims_extra=2, max_side=10),)
def test_gf_form_moments(args):
    """Check that the Gfs constructed from moments have the correct moment."""
    mom, = args
    gf = gt.basis.pole.gf_from_moments(mom, width=1)
    gf_mom = gf.moments(np.arange(mom.shape[-1])+1)
    assert_allclose(mom, gf_mom, equal_nan=True, atol=1e-12)


def test_gf_form_moments_nan():
    """Check that the Gfs constructed from moments handle NaN."""
    mom = [np.nan]
    gf = gt.basis.pole.gf_from_moments(mom)
    gf_mom = gt.pole_gf_moments(poles=gf.poles, weights=gf.residues, order=1)
    assert_allclose(mom, gf_mom, equal_nan=True)


@given(pole=st.floats(-10, 10))  # necessary Matsubaras depend on Bandwidth!
def test_iw2tau_dft_soft_single_pole(pole):
    """Low accuracy test of `iw2tau_dft_soft` on a single pole."""
    BETA = 1.3
    N_IW = 4096
    iws = gt.matsubara_frequencies(range(N_IW), beta=BETA)
    tau = np.linspace(0, BETA, num=2*N_IW+1)

    gf_iw = gt.pole_gf_z(iws, poles=[pole], weights=[1.])
    gf_dft = gt.fourier.iw2tau_dft_soft(gf_iw - 1./iws, beta=BETA) - .5
    gf_tau = gt.pole_gf_tau(tau, poles=[pole], weights=[1.], beta=BETA)

    assert_allclose(gf_tau, gf_dft, atol=2e-3, rtol=2e-4)


@given(pole=st.floats(-10, 10))  # necessary Matsubaras depend on Bandwidth!
def test_iw2tau_dft_single_pole(pole):
    """Low accuracy test of `iw2tau_dft` on a single pole."""
    BETA = 1.3
    N_IW = 4096
    iws = gt.matsubara_frequencies(range(N_IW), beta=BETA)
    tau = np.linspace(0, BETA, num=2*N_IW+1)

    gf_iw = gt.pole_gf_z(iws, poles=[pole], weights=[1.])
    gf_dft = gt.fourier.iw2tau_dft(gf_iw - 1./iws, beta=BETA) - .5
    gf_tau = gt.pole_gf_tau(tau, poles=[pole], weights=[1.], beta=BETA)

    assert_allclose(gf_tau, gf_dft, atol=1e-3, rtol=1e-4)


@given(gufunc_args('(n),(n)->(l)', dtype=np.float_,
                   elements=[st.floats(min_value=-10, max_value=10),
                             st.floats(min_value=0, max_value=10)],
                   max_dims_extra=1, max_side=5),)
def test_iw2tau_multi_pole(args):
    """Test `iw2tau` for a multi-pole Green's function."""
    poles, resids = args
    assume(np.all(resids.sum(axis=-1) > 1e-4))
    resids /= resids.sum(axis=-1, keepdims=True)  # not really necessary...
    m0 = resids.sum(axis=-1, keepdims=True)
    BETA = 1.3
    N_IWS = 1024
    iws = gt.matsubara_frequencies(range(N_IWS), beta=BETA)
    tau = np.linspace(0, BETA, num=2*N_IWS + 1)

    gf_iw = gt.pole_gf_z(iws, poles=poles[..., np.newaxis, :], weights=resids[..., np.newaxis, :])

    gf_ft = gt.fourier.iw2tau(gf_iw, beta=BETA)
    gf_dft = gt.fourier.iw2tau_dft(gf_iw - m0/iws, beta=BETA) - m0*.5
    # without additional information tau2iw should match back-end
    assert_allclose(gf_ft, gf_dft)

    gf_tau = gt.pole_gf_tau(tau, poles=poles[..., np.newaxis, :],
                            weights=resids[..., np.newaxis, :], beta=BETA)
    # using many moments should give exact result
    mom = gt.pole_gf_moments(poles, resids, order=range(1, 6))
    gf_ft = gt.fourier.iw2tau(gf_iw, moments=mom, beta=BETA)
    assert_allclose(gf_tau, gf_ft, atol=1e-10)
    gf_ft = gt.fourier.iw2tau(gf_iw, moments=mom[..., :2], beta=BETA, n_fit=2)
    assert_allclose(gf_tau, gf_ft, atol=1e-10)


# TODO: check if there is a way to improve result for pole↘0
@pytest.mark.filterwarnings("ignore:(overflow):RuntimeWarning")
@given(pole=st.floats(min_value=1e-12, exclude_min=True, allow_infinity=False))
def test_tau2iv_ft_lin_single_pole(pole):
    """Low accuracy test of `tau2iv_ft_lin` on a single pole."""
    BETA = 1.3
    N_TAU = 4096 + 1
    tau = np.linspace(0, BETA, num=N_TAU)
    ivs = gt.matsubara_frequencies_b(range(N_TAU//2 + 1), beta=BETA)

    gf_tau = gt.pole_gf_tau_b(tau, poles=[pole], weights=[1.], beta=BETA)
    gf_ft_lin = gt.fourier.tau2iv_ft_lin(gf_tau, beta=BETA)
    gf_iv = gt.pole_gf_z(ivs, poles=[pole], weights=[1.])

    assert_allclose(gf_iv, gf_ft_lin, atol=2e-4)


@pytest.mark.filterwarnings("ignore:(overflow):RuntimeWarning")
@given(pole=st.floats(min_value=1e-2, exclude_min=True, allow_infinity=False))
def test_tau2iv_single_pole(pole):
    """Test that `tau2iv` improves plain results on a single pole."""
    BETA = 1.3
    N_TAU = 4096 + 1
    tau = np.linspace(0, BETA, num=N_TAU)
    ivs = gt.matsubara_frequencies_b(range(N_TAU//2 + 1), beta=BETA)

    gf_tau = gt.pole_gf_tau_b(tau, poles=[pole], weights=[1.], beta=BETA)
    gf_dft = gt.fourier.tau2iv(gf_tau, beta=BETA, fourier=gt.fourier.tau2iv_dft)
    gf_dft_bare = gt.fourier.tau2iv_dft(gf_tau, beta=BETA)
    gf_iv = gt.pole_gf_z(ivs, poles=[pole], weights=[1.])

    err = abs(gf_iv - gf_dft)
    err_bare = abs(gf_iv - gf_dft_bare)
    assert np.all((err <= err_bare) | np.isclose(err, err_bare, atol=1e-8))


@pytest.mark.filterwarnings("ignore:(overflow):RuntimeWarning")
@given(pole=st.floats(allow_nan=False, allow_infinity=False))
def test_tau2iw_ft_lin_single_pole(pole):
    """Low accuracy test of `tau2iw_ft_lin` on a single pole."""
    BETA = 1.3
    N_TAU = 4096 + 1
    tau = np.linspace(0, BETA, num=N_TAU)
    iws = gt.matsubara_frequencies(range(N_TAU//2), beta=BETA)

    gf_tau = gt.pole_gf_tau(tau, poles=[pole], weights=[1.], beta=BETA)
    gf_ft_lin = gt.fourier.tau2iw_ft_lin(gf_tau + .5, beta=BETA) + 1/iws
    gf_iw = gt.pole_gf_z(iws, poles=[pole], weights=[1.])

    assert_allclose(gf_iw, gf_ft_lin, atol=2e-4)


@pytest.mark.filterwarnings("ignore:(overflow):RuntimeWarning")
@given(pole=st.floats(allow_nan=False, allow_infinity=False))
def test_tau2iw_dft_single_pole(pole):
    """Low accuracy test of `tau2iw_dft` on a single pole."""
    BETA = 1.3
    N_TAU = 8192 + 1
    tau = np.linspace(0, BETA, num=N_TAU)
    iws = gt.matsubara_frequencies(range(N_TAU//2), beta=BETA)

    gf_tau = gt.pole_gf_tau(tau, poles=[pole], weights=[1.], beta=BETA)
    gf_dft = gt.fourier.tau2iw_dft(gf_tau + .5, beta=BETA) + 1/iws
    gf_iw = gt.pole_gf_z(iws, poles=[pole], weights=[1.])

    assert_allclose(gf_iw, gf_dft, atol=1e-4)


@given(gufunc_args('(n),(n)->(l)', dtype=np.float_,
                   elements=[st.floats(min_value=-10, max_value=10),
                             st.floats(min_value=0, max_value=10)],
                   max_dims_extra=1, max_side=5),)
def test_tau2iw_multi_pole(args):
    """Test `tau2iw` for a multi-pole Green's function."""
    poles, resids = args
    assume(np.all(resids.sum(axis=-1) > 1e-4))
    resids /= resids.sum(axis=-1, keepdims=True)
    m0 = resids.sum(axis=-1, keepdims=True)
    BETA = 1.3
    N_TAU = 2048 + 1
    tau = np.linspace(0, BETA, num=N_TAU)
    iws = gt.matsubara_frequencies(range(N_TAU//2), beta=BETA)

    gf_tau = gt.pole_gf_tau(tau, poles=poles[..., np.newaxis, :],
                            weights=resids[..., np.newaxis, :], beta=BETA)
    gf_ft = gt.fourier.tau2iw(gf_tau, beta=BETA)

    gf_ft_lin = gt.fourier.tau2iw_ft_lin(gf_tau + m0*.5, beta=BETA) + m0/iws
    # without additional information tau2iw should match back-end
    assert_allclose(gf_ft, gf_ft_lin)
    gf_iw = gt.pole_gf_z(iws, poles=poles[..., np.newaxis, :], weights=resids[..., np.newaxis, :])

    # fitting many poles it should get very good
    gf_ft = gt.fourier.tau2iw(gf_tau, n_pole=25, beta=BETA)
    assert_allclose(gf_iw, gf_ft, rtol=1e-4)


@given(gufunc_args('(n),(n)->(l)', dtype=np.float_,
                   elements=[st.floats(min_value=-10, max_value=10),
                             st.floats(min_value=0, max_value=10)],
                   max_dims_extra=1, max_side=5),)
def test_tau2iw_multi_pole_hfm(args):
    """Test `tau2iw_dft` for a multi-pole Green's function."""
    poles, resids = args
    assume(np.all(resids.sum(axis=-1) > 1e-4))
    resids /= resids.sum(axis=-1, keepdims=True)
    BETA = 1.3
    N_TAU = 2048 + 1
    tau = np.linspace(0, BETA, num=N_TAU)
    iws = gt.matsubara_frequencies(range(N_TAU//2), beta=BETA)
    gf_tau = gt.pole_gf_tau(tau, poles=poles[..., np.newaxis, :],
                            weights=resids[..., np.newaxis, :], beta=BETA)
    # fitting a few high-frequency moments shouldn't hurt either
    # -> actually it is rather bad...
    mom = gt.pole_gf_moments(poles, resids, order=range(1, 3))
    gf_ft = gt.fourier.tau2iw(gf_tau, n_pole=25, moments=mom, beta=BETA)
    gf_iw = gt.pole_gf_z(iws, poles=poles[..., np.newaxis, :], weights=resids[..., np.newaxis, :])
    assert_allclose(gf_iw, gf_ft, rtol=1e-4)


@given(gufunc_args('(n)->(n)', dtype=np.float_,
                   elements=st.floats(min_value=-1e6, max_value=1e6),
                   max_dims_extra=2, max_side=10),)
def test_pole_from_gftau_exact(args):
    """Recover exact residues from Pole Gf with Chebyshev poles."""
    resids, = args
    n_poles = resids.shape[-1]
    assume(n_poles > 0)
    poles = np.cos(.5*np.pi*np.arange(1, 2*n_poles, 2)/n_poles)[::-1]
    beta = 13.78
    tau = np.linspace(0, beta, num=1024)
    gf_tau = gt.pole_gf_tau(tau, poles=poles, weights=resids[..., np.newaxis, :], beta=beta)
    pole_gf = gt.basis.pole.gf_from_tau(gf_tau, n_poles, beta=beta)
    assert_allclose(pole_gf.poles, poles, atol=1e-12)
    try:
        atol = max(1e-8, abs(resids).max())
    except ValueError:
        atol = 1e-8
    assert_allclose(pole_gf.residues, resids, atol=atol)


@pytest.fixture(scope="module")
def pade_frequencies():
    """Provide Padé frequency as they are slow to calculate."""
    izp, rp = gt.pade_frequencies(20, beta=1)
    izp.flags.writeable = False
    rp.flags.writeable = False

    def pade_frequencies_(beta):
        return izp / beta, rp

    return pade_frequencies_


@given(poles=arrays(float, 10, elements=st.floats(-10, 10)),
       resids=arrays(float, 10, elements=st.floats(0, 10)))
def test_izp2tau_multi_pole(poles, resids, pade_frequencies):
    """Test `izp2tau` for a multi-pole Green's function."""
    assume(np.all(resids.sum(axis=-1) > 1e-4))
    resids /= resids.sum(axis=-1, keepdims=True)  # not really necessary...
    BETA = 1.7
    izp, __ = pade_frequencies(BETA)
    tau = np.linspace(0, BETA, num=1025)

    gf_pole = gt.basis.PoleGf(poles=poles, residues=resids)
    gf_izp = gf_pole.eval_z(izp)

    gf_ft = gt.fourier.izp2tau(izp, gf_izp, tau=tau, beta=BETA)
    gf_tau = gf_pole.eval_tau(tau, beta=BETA)
    assert_allclose(gf_tau, gf_ft)

    # using many moments should give exact result
    mom = gf_pole.moments(order=range(1, 4))
    gf_ft = gt.fourier.izp2tau(izp, gf_izp, tau=tau, beta=BETA, moments=mom)
    assert_allclose(gf_tau, gf_ft)


@given(poles=arrays(float, 10, elements=st.floats(-10, 10)),
       resids=arrays(float, 10, elements=st.floats(0, 10)))
def test_tau2izp_multi_pole(poles, resids, pade_frequencies):
    """Test `tau2izp` for a multi-pole Green's function."""
    assume(np.all(resids.sum(axis=-1) > 1e-4))
    resids /= resids.sum(axis=-1, keepdims=True)  # not really necessary...
    BETA = 1.7
    izp, __ = pade_frequencies(BETA)  # should perform badly for few frequencies...
    tau = np.linspace(0, BETA, num=129)

    gf_pole = gt.basis.PoleGf(poles=poles, residues=resids)
    gf_tau = gf_pole.eval_tau(tau, beta=BETA)

    gf_ft = gt.fourier.tau2izp(gf_tau, BETA, izp)
    gf_izp = gf_pole.eval_z(izp)
    assert_allclose(gf_izp, gf_ft)

    # using many moments should give exact result
    mom = gf_pole.moments(order=range(1, 4))
    occ = gf_pole.occ(BETA)
    gf_ft = gt.fourier.tau2izp(gf_tau, BETA, izp, moments=mom, occ=occ)

    assert_allclose(gf_izp, gf_ft, rtol=1e-5, atol=1e-8)

    # check why atol is necessary, example below (seems empty)
    # poles=array([-8.84053211, -8.84053211, -8.84053211, -8.84053211, -8.84053211,
    #        -8.84053211, -8.84053211, -8.84053211, -8.84053211, -8.84053211]),
    # resids=array([8.11071633, 8.11071633, 8.11071633, 8.11071633, 3.29058834,
    #        3.11249264, 8.11071633, 8.11071633, 8.11071633, 8.11071633]),


@pytest.mark.parametrize("test_fct", [
    partial(gt.lattice.bethe.gf_ret_t, half_bandwidth=1, center=0.2),
    np.ones_like,
])
def test_trapz_weights(test_fct):
    """Test weights for trapezoidal rule."""
    tt, dt = np.linspace(0, 10, 101, retstep=True)
    gf = test_fct(tt)
    coeff = gt.fourier._trapz_weight(dt, gf_t=gf, endpoint=True)
    assert_allclose(coeff.sum(), np.trapz(gf, dx=dt), rtol=1e-14)
    gf[-1] = 0
    coeff = gt.fourier._trapz_weight(dt, gf_t=gf, endpoint=False)
    assert_allclose(coeff.sum(), np.trapz(gf, dx=dt), rtol=1e-14)


@pytest.mark.parametrize("test_fct", [
    partial(gt.lattice.bethe.gf_ret_t, half_bandwidth=1, center=0.2),
    np.ones_like,
])
def test_simps_weights(test_fct):
    """Test weights for Simpson rule."""
    for num in [100, 101]:  # test even as well as odd!
        tt, dt = np.linspace(0, 10, num, retstep=True)
        gf = test_fct(tt)
        coeff = gt.fourier._simps_weight(dt, gf_t=gf, endpoint=True)
        assert_allclose(coeff.sum(), simpson(gf, dx=dt, even='first'), rtol=1e-14)
        gf[-2:] = 0
        coeff = gt.fourier._trapz_weight(dt, gf_t=gf, endpoint=False)
        assert_allclose(coeff.sum(), np.trapz(gf, dx=dt), rtol=1e-14)


@given(gufunc_args('(n),(n)->(l)', dtype=np.float_,
                   elements=[st.floats(min_value=-10, max_value=10),
                             st.floats(min_value=0, max_value=10), ],
                   max_dims_extra=2, max_side=5),)
def test_tt2z_trapz_naive_gubehaviour(args):
    """Compare optimized to naive trapezoidal rule."""
    poles, resids = args
    tt = np.linspace(0, 10, num=101)
    ww = np.linspace(-5, 5, num=57) + 0.1j
    gf_t = gt.pole_gf_ret_t(tt, poles=poles[..., np.newaxis, :], weights=resids[..., np.newaxis, :])
    gf_ft = gt.fourier.tt2z_trapz(tt, gf_t, ww)
    naiv = np.trapz(np.exp(1j*ww[:, None]*tt)*gf_t[..., None, :], x=tt)
    assert_allclose(gf_ft, naiv, rtol=1e-12, atol=1e-14)


@given(spole=st.floats(-1, 1))  # oscillation speed depends on bandwidth
def test_tt2z_single_pole(spole):
    """Low accuracy test of `tt2z` on a single pole."""
    tt = np.linspace(0, 50, 3001)
    ww = np.linspace(-2, 2, num=101) + 2e-1j

    gf_t = gt.pole_gf_ret_t(tt, poles=[spole], weights=[1.])
    gf_z = gt.pole_gf_z(ww, poles=[spole], weights=[1.])

    gf_dft = gt.fourier.tt2z(tt, gf_t=gf_t, z=ww, laplace=gt.fourier.tt2z_trapz)
    assert_allclose(gf_z, gf_dft, atol=2e-3, rtol=2e-4)
    gf_dft = gt.fourier.tt2z(tt, gf_t=gf_t, z=ww, laplace=gt.fourier.tt2z_simps)
    assert_allclose(gf_z, gf_dft, atol=2e-3, rtol=2e-4)
    gf_dft = gt.fourier.tt2z(tt, gf_t=gf_t, z=ww, laplace=gt.fourier.tt2z_lin)
    assert_allclose(gf_z, gf_dft, atol=1e-3, rtol=2e-4)


@given(spole=st.floats(-1, 1))  # oscillation speed depends on bandwidth
@pytest.mark.parametrize("num", [3, 4, 100])  # test for even and odd numbers and overfitting
def test_tt2z_pade_single_pole(spole, num):
    """Test of `tt2z_pade` on a single pole."""
    dt = 0.01  # discretization determines error
    tt = np.linspace(0, dt*(num - 1), num)
    ww = np.linspace(-1.5, 1.5, num=201) + 1e-4j

    gf_t = gt.pole_gf_ret_t(tt, poles=[spole], weights=[1.])
    gf_z = gt.pole_gf_z(ww, poles=[spole], weights=[1.])

    gf_pf = gt.fourier.tt2z(tt, gf_t=gf_t, z=ww, laplace=gt.fourier.tt2z_pade)
    assert_allclose(gf_z, gf_pf, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not gt.fourier._HAS_NUMEXPR,
                    reason="Only fall-back available, already tested.")
@given(spole=st.floats(-1, 1))  # oscillation speed depends on bandwidth
def test_tt2z_single_pole_nonumexpr(spole):
    """Low accuracy test of `tt2z` on a single pole."""
    tt = np.linspace(0, 50, 1001)
    ww = np.linspace(-2, 2, num=51) + 2e-1j

    gf_t = gt.pole_gf_ret_t(tt, poles=[spole], weights=[1.])
    gf_z = gt.pole_gf_z(ww, poles=[spole], weights=[1.])

    try:
        gt.fourier._phase = gt.fourier._phase_numpy
        gt.fourier._HAS_NUMEXPR = False
        gf_dft = gt.fourier.tt2z(tt, gf_t=gf_t, z=ww, laplace=gt.fourier.tt2z_trapz)
        assert_allclose(gf_z, gf_dft, atol=3e-3, rtol=3e-4)
        gf_dft = gt.fourier.tt2z(tt, gf_t=gf_t, z=ww, laplace=gt.fourier.tt2z_simps)
        assert_allclose(gf_z, gf_dft, atol=3e-3, rtol=3e-4)
        gf_dft = gt.fourier.tt2z(tt, gf_t=gf_t, z=ww, laplace=gt.fourier.tt2z_lin)
        assert_allclose(gf_z, gf_dft, atol=2e-3, rtol=3e-4)
    finally:
        gt.fourier._phase = gt.fourier._phase_numexpr


@given(gufunc_args('(n),(n)->(l)', dtype=np.float_,
                   elements=[st.floats(min_value=-1, max_value=1),
                             st.floats(min_value=0, max_value=10), ],
                   max_dims_extra=2, max_side=5),)
def test_tt2z_multi_pole(args):
    """Test `tt2z` for a multi-pole Green's function."""
    poles, resids = args
    assume(np.all(resids.sum(axis=-1) > 1e-4))
    resids /= resids.sum(axis=-1, keepdims=True)
    tt = np.linspace(0, 50, 3001)
    ww = np.linspace(-2, 2, num=101) + 2e-1j

    gf_t = gt.pole_gf_ret_t(tt, poles=poles[..., np.newaxis, :],
                            weights=resids[..., np.newaxis, :])
    gf_z = gt.pole_gf_z(ww, poles=poles[..., np.newaxis, :],
                        weights=resids[..., np.newaxis, :])

    gf_ft = gt.fourier.tt2z(tt, gf_t, ww, laplace=gt.fourier.tt2z_lin)
    assert_allclose(gf_z, gf_ft, rtol=1e-3)

    gf_ft = gt.fourier.tt2z(tt, gf_t, ww, laplace=gt.fourier.tt2z_trapz)
    assert_allclose(gf_z, gf_ft, rtol=1e-3)

    gf_ft = gt.fourier.tt2z(tt, gf_t, ww, laplace=gt.fourier.tt2z_simps)
    assert_allclose(gf_z, gf_ft, rtol=1e-3)

    # test if zero handles gu-structure correctly
    ww[ww.size//2] = 0
    gt.fourier.tt2z(tt[::10], gf_t[..., ::10], ww, laplace=gt.fourier.tt2z_lin)
    ww[0] = ww[-1] = 0
    gt.fourier.tt2z(tt[::10], gf_t[..., ::10], ww, laplace=gt.fourier.tt2z_lin)


@given(gufunc_args('(l),(n),(n)->(l)', dtype=np.complex_,
                   elements=[st.complex_numbers(max_magnitude=2),
                             st.floats(min_value=-1, max_value=1),
                             st.floats(min_value=0, max_value=10), ],
                   max_dims_extra=2, max_side=5),)
def test_tt2z_gufuncz(args):
    """Test `tt2z` for different shapes of `z`."""
    z, poles, resids = args
    assume(np.all(resids.sum(axis=-1) > 1e-4))
    resids /= resids.sum(axis=-1, keepdims=True)
    # ensure sufficient large imaginary part
    z = np.where(z.imag < 0, z.conj(), z)
    z += 2e-1j
    tt = np.linspace(0, 50, 3001)

    gf_t = gt.pole_gf_ret_t(tt, poles=poles[..., np.newaxis, :],
                            weights=resids[..., np.newaxis, :])
    gf_z = gt.pole_gf_z(z, poles=poles[..., np.newaxis, :],
                        weights=resids[..., np.newaxis, :])

    gf_ft = gt.fourier.tt2z(tt, gf_t, z, laplace=gt.fourier.tt2z_lin)
    assert_allclose(gf_z, gf_ft, rtol=1e-3)

    gf_ft = gt.fourier.tt2z(tt, gf_t, z, laplace=gt.fourier.tt2z_trapz)
    assert_allclose(gf_z, gf_ft, rtol=1e-3)

    gf_ft = gt.fourier.tt2z(tt, gf_t, z, laplace=gt.fourier.tt2z_simps)
    assert_allclose(gf_z, gf_ft, rtol=1e-3)


@given(gufunc_args('(l),(n),(n)->(l)', dtype=np.complex_,
                   elements=[st.floats(min_value=-1,  max_value=1),
                             st.floats(min_value=-1, max_value=1),
                             st.floats(min_value=0, max_value=10), ],
                   max_dims_extra=2, max_side=3),)
def test_tt2z_gufuncz_pade(args):
    """Test `tt2z_pade` for different shapes of `z`."""
    ww, poles, resids = args
    assume(np.all(resids.sum(axis=-1) > 1e-4))
    resids /= resids.sum(axis=-1, keepdims=True)
    # ensure sufficient large imaginary part
    z = ww + 1e-3j
    tt = np.linspace(0, 10, 201)

    gf_t = gt.pole_gf_ret_t(tt, poles=poles[..., np.newaxis, :],
                            weights=resids[..., np.newaxis, :])
    gf_z = gt.pole_gf_z(z, poles=poles[..., np.newaxis, :],
                        weights=resids[..., np.newaxis, :])

    gf_pf = gt.fourier.tt2z(tt, gf_t, z, laplace=gt.fourier.tt2z_pade)
    assert_allclose(gf_z, gf_pf, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("fast", [False, True])
def test_tt2z_pade_bethe(fast):
    """Test `tt2z_pade` against Bethe Green's function."""
    z = np.linspace(-2, 2, num=1001) + 1e-3j
    tt = np.linspace(0, 20, 201)
    D = 1.5
    mu = 0.2

    gf_t = gt.lattice.bethe.gf_ret_t(tt, half_bandwidth=D, center=-mu)
    gf_ww = gt.lattice.bethe.gf_z(z + mu*D, half_bandwidth=D)
    gf_fp = gt.fourier.tt2z(tt, gf_t, z=z, laplace=gt.fourier.tt2z_pade, fast=fast)

    # error should be local to the band edges
    inner = (-1.5 < z.real) & (z.real < 1)
    assert_allclose(gf_fp[inner], gf_ww[inner], rtol=2e-3 if fast else 1e-3)
    assert_allclose(gf_fp[~inner], gf_ww[~inner], rtol=0.05)


@pytest.mark.parametrize("fast", [False, True])
def test_tt2z_pade_box(fast):
    """Test `tt2z_pade` against box Green's function."""
    z = np.linspace(-2, 2, num=1001) + 1e-3j
    tt = np.linspace(0, 20, 201)
    D = 1.5
    mu = 0.2

    gf_t = gt.lattice.box.gf_ret_t(tt, half_bandwidth=D, center=-mu)
    gf_ww = gt.lattice.box.gf_z(z + mu*D, half_bandwidth=D)
    gf_fp = gt.fourier.tt2z(tt, gf_t, z=z, laplace=gt.fourier.tt2z_pade, fast=fast)

    # error should be local to the band edges
    inner = (-1.5 < z.real) & (z.real < 0.9)
    assert_allclose(gf_fp[inner], gf_ww[inner], rtol=1e-3)
    assert_allclose(gf_fp[~inner], gf_ww[~inner], rtol=0.3 if fast else 0.2)


@given(gufunc_args('(l),(n),(n)->(l)', dtype=np.complex_,
                   elements=[st.floats(min_value=-1,  max_value=1),
                             st.floats(min_value=-1, max_value=1),
                             st.floats(min_value=0, max_value=10), ],
                   max_dims_extra=2, max_side=3),)
def test_tt2z_gufuncz_lpz(args):
    """Test `tt2z_lpz` for different shapes of `z`."""
    ww, poles, resids = args
    assume(np.all(resids.sum(axis=-1) > 1e-4))
    resids /= resids.sum(axis=-1, keepdims=True)
    # ensure sufficient large imaginary part
    z = ww + 1e-3j
    tt = np.linspace(0, 10, 201)

    gf_t = gt.pole_gf_ret_t(tt, poles=poles[..., np.newaxis, :],
                            weights=resids[..., np.newaxis, :])
    gf_z = gt.pole_gf_z(z, poles=poles[..., np.newaxis, :],
                        weights=resids[..., np.newaxis, :])

    gf_pf = gt.fourier.tt2z(tt, gf_t, z, laplace=gt.fourier.tt2z_lpz)
    assert_allclose(gf_z, gf_pf, rtol=1e-5, atol=1e-3)


def test_tt2z_lpz_bethe():
    """Test `tt2z_lpz` against Bethe Green's function."""
    z = np.linspace(-2, 2, num=1001) + 1e-3j
    tt = np.linspace(0, 20, 201)
    D = 1.5
    mu = 0.2

    gf_t = gt.lattice.bethe.gf_ret_t(tt, half_bandwidth=D, center=-mu)
    gf_ww = gt.lattice.bethe.gf_z(z + mu*D, half_bandwidth=D)
    gf_fp = gt.fourier.tt2z(tt, gf_t, z=z, laplace=gt.fourier.tt2z_lpz)

    # error should be local to the band edges
    inner = (-1.5 < z.real) & (z.real < 1)
    assert_allclose(gf_fp[inner], gf_ww[inner], rtol=1e-3)
    assert_allclose(gf_fp[~inner], gf_ww[~inner], rtol=0.05)


def test_tt2z_lpz_box():
    """Test `tt2z_lpz` against box Green's function."""
    z = np.linspace(-2, 2, num=1001) + 1e-3j
    tt = np.linspace(0, 20, 201)
    D = 1.5
    mu = 0.2

    gf_t = gt.lattice.box.gf_ret_t(tt, half_bandwidth=D, center=-mu)
    gf_ww = gt.lattice.box.gf_z(z + mu*D, half_bandwidth=D)
    gf_fp = gt.fourier.tt2z(tt, gf_t, z=z, laplace=gt.fourier.tt2z_lpz)

    # error should be local to the band edges
    inner = (-1.5 < z.real) & (z.real < 0.9)
    assert_allclose(gf_fp[inner], gf_ww[inner], rtol=1e-3)
    assert_allclose(gf_fp[~inner], gf_ww[~inner], rtol=0.2)


def test_tt2z_pader_bethe():
    """Test robust `tt2z_pade` against Bethe Green's function."""
    z = np.linspace(-2, 2, num=1001) + 1e-3j
    tt = np.linspace(0, 50, 501)
    D = 1.5
    mu = 0.2

    noise = np.random.default_rng(0).normal(scale=1e-6, size=tt.size)
    gf_t = gt.lattice.bethe.gf_ret_t(tt, half_bandwidth=D, center=-mu)
    gf_ww = gt.lattice.bethe.gf_z(z + mu*D, half_bandwidth=D)
    gf_fp = gt.fourier.tt2z(tt, gf_t+noise, z=z, laplace=gt.fourier.tt2z_pade,
                            pade=gt.hermpade.pader, rcond=1e-8)
    # error should be local to the band edges
    inner = (-1.5 < z.real) & (z.real < 1)
    assert_allclose(gf_fp[inner], gf_ww[inner], rtol=1e-3)
    assert_allclose(gf_fp[~inner], gf_ww[~inner], rtol=0.05)


@pytest.mark.parametrize("num", [  # test for minimum, and overfitting
    5, 6, 7,  # minimum numbers, 3 steps as `degree = size//3`
    pytest.param(100, marks=pytest.mark.xfail(
        reason="CI pipeline fails this test, I am however unable to locally reproduce it"
    )),
    500,  # large overfitting to check stability
])
def test_tt2z_herm2_bethe(num):
    """Test `tt2z_herm2` against Bethe Green's function."""
    z = np.linspace(-2, 2, num=1001) + 1e-4j
    dt = 0.01  # discretization determines error
    tt = np.linspace(0, dt*(num - 1), num)
    D = 1.5
    mu = 0.2

    gf_t = gt.lattice.bethe.gf_ret_t(tt, half_bandwidth=D, center=-mu)
    gf_ww = gt.lattice.bethe.gf_z(z + mu*D, half_bandwidth=D)
    gf_fh = gt.fourier.tt2z(tt, gf_t, z=z, laplace=gt.fourier.tt2z_herm2)
    gf_fp = gt.fourier.tt2z(tt, gf_t, z=z, laplace=gt.fourier.tt2z_pade)

    # approximation is good globally
    assert_allclose(gf_fh, gf_ww, rtol=2e-4, atol=2e-4)
    # better than Padé
    assert np.linalg.norm(gf_ww - gf_fh) < np.linalg.norm(gf_ww - gf_fp)


@pytest.mark.parametrize("herm, otol", [
    (gt.hermpade.Hermite2.from_taylor, 0.05),
    (gt.hermpade._Hermite2Ret.from_taylor, 0.05),
    (gt.hermpade.Hermite2.from_taylor_lstsq, 0.06),
    (gt.hermpade._Hermite2Ret.from_taylor_lstsq, 0.06),
])
def test_tt2z_herm2_box(herm, otol):
    """Test `tt2z_herm2` against box Green's function."""
    z = np.linspace(-2, 2, num=1001) + 1e-3j
    tt = np.linspace(0, 20, 201)
    D = 1.5
    mu = 0.2

    gf_t = gt.lattice.box.gf_ret_t(tt, half_bandwidth=D, center=-mu)
    gf_ww = gt.lattice.box.gf_z(z + mu*D, half_bandwidth=D)
    gf_fh = gt.fourier.tt2z(tt, gf_t, z=z, laplace=gt.fourier.tt2z_herm2, herm2=herm)
    gf_fp = gt.fourier.tt2z(tt, gf_t, z=z, laplace=gt.fourier.tt2z_pade)

    # error should be local to the band edges
    inner = (-1.5 < z.real) & (z.real < 1)
    assert_allclose(gf_fh[inner], gf_ww[inner], rtol=1e-3)
    assert_allclose(gf_fh[~inner], gf_ww[~inner], rtol=otol)
    # better than Padé
    assert np.linalg.norm(gf_ww - gf_fh) < np.linalg.norm(gf_ww - gf_fp)
