# coding: utf8
"""Test the Fourier transformation of Green's functions."""
import numpy as np
import pytest
import hypothesis.strategies as st

from hypothesis import given, assume
from hypothesis_gufunc.gufunc import gufunc_args

from .context import gftool as gt
from .context import pole


@given(gufunc_args('(n)->(n),(m)', dtype=np.float_,
                   elements=st.floats(min_value=-1e6, max_value=1e6),
                   max_dims_extra=2, max_side=10),)
def test_gf_form_moments(args):
    """Check that the Gfs constructed from moments have the correct moment."""
    mom, = args
    gf = pole.gf_from_moments(mom, width=1)
    gf_mom = gf.moments(np.arange(mom.shape[-1])+1)
    assert np.allclose(mom, gf_mom, equal_nan=True)


def test_gf_form_moments_nan():
    """Check that the Gfs constructed from moments handle NaN."""
    mom = [np.nan]
    gf = pole.gf_from_moments(mom)
    gf_mom = gt.pole_gf_moments(poles=gf.poles, weights=gf.residues, order=1)
    assert np.allclose(mom, gf_mom, equal_nan=True)


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

    assert np.allclose(gf_tau, gf_dft, atol=2e-3, rtol=2e-4)


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

    assert np.allclose(gf_tau, gf_dft, atol=1e-3, rtol=1e-4)


@given(gufunc_args('(n),(n)->(l)', dtype=np.float_,
                   elements=[st.floats(min_value=-10, max_value=10),
                             st.floats(min_value=0, max_value=10),],
                   max_dims_extra=1, max_side=5),)
def test_iw2tau_mulity_pole(args):
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
    assert np.allclose(gf_ft, gf_dft)

    gf_tau = gt.pole_gf_tau(tau, poles=poles[..., np.newaxis, :],
                            weights=resids[..., np.newaxis, :], beta=BETA)
    # using many moments should give exact result
    mom = gt.pole_gf_moments(poles, resids, order=range(1, 6))
    gf_ft = gt.fourier.iw2tau(gf_iw, moments=mom, beta=BETA)
    assert np.allclose(gf_tau, gf_ft)
    gf_ft = gt.fourier.iw2tau(gf_iw, moments=mom[..., :2], beta=BETA, n_fit=2)
    assert np.allclose(gf_tau, gf_ft)


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

    assert np.allclose(gf_iv, gf_ft_lin, atol=2e-4)


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

    assert np.allclose(gf_iw, gf_ft_lin, atol=2e-4)


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

    assert np.allclose(gf_iw, gf_dft, atol=1e-4)


@given(gufunc_args('(n),(n)->(l)', dtype=np.float_,
                   elements=[st.floats(min_value=-10, max_value=10),
                             st.floats(min_value=0, max_value=10),],
                   max_dims_extra=1, max_side=5),)
def test_tau2iw_mulity_pole(args):
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
    assert np.allclose(gf_ft, gf_ft_lin)
    gf_iw = gt.pole_gf_z(iws, poles=poles[..., np.newaxis, :], weights=resids[..., np.newaxis, :])

    # fitting many poles it should get very good
    gf_ft = gt.fourier.tau2iw(gf_tau, n_pole=25, beta=BETA)
    assert np.allclose(gf_iw, gf_ft, rtol=1e-4)


@given(gufunc_args('(n),(n)->(l)', dtype=np.float_,
                   elements=[st.floats(min_value=-10, max_value=10),
                             st.floats(min_value=0, max_value=10),],
                   max_dims_extra=1, max_side=5),)
def test_tau2iw_mulity_pole_hfm(args):
    """Test `tau2iw_dft` for a multi-pole Green's function."""
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
    # fitting a few high-frequency moments shouldn't hurt either
    # -> actually it is rather bad...
    mom = gt.pole_gf_moments(poles, resids, order=range(1, 3))
    gf_ft = gt.fourier.tau2iw(gf_tau, n_pole=25, moments=mom, beta=BETA)
    gf_iw = gt.pole_gf_z(iws, poles=poles[..., np.newaxis, :], weights=resids[..., np.newaxis, :])
    assert np.allclose(gf_iw, gf_ft, rtol=1e-4)


@given(gufunc_args('(n)->(n)', dtype=np.float_,
                   elements=st.floats(min_value=-1e6, max_value=1e6),
                   max_dims_extra=2, max_side=10),)
def test_pole_from_gftau_exact(args):
    """Recover exact residues from Pole Gf with Chebyshev poles."""
    resids, = args
    n_poles = resids.shape[-1]
    poles = np.cos(.5*np.pi*np.arange(1, 2*n_poles, 2)/n_poles)
    beta = 13.78
    tau = np.linspace(0, beta, num=1024)
    gf_tau = gt.pole_gf_tau(tau, poles=poles, weights=resids[..., np.newaxis, :], beta=beta)
    pole_gf = pole.gf_from_tau(gf_tau, n_poles, beta=beta)
    assert np.allclose(pole_gf.poles, poles)
    try:
        atol = max(1e-8, abs(resids).max())
    except ValueError:
        atol = 1e-8
    assert np.allclose(pole_gf.residues, resids, atol=atol)
