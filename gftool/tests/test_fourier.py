# coding: utf8
"""Test the Fourier transformation of Green's functions."""
import numpy as np
import hypothesis.strategies as st

from hypothesis import given
from hypothesis_gufunc.gufunc import gufunc_args

from .context import gftool as gt


@given(gufunc_args('(n)->(n),(m)', dtype=np.float_,
                   elements=st.floats(min_value=-1e6, max_value=1e6),
                   max_dims_extra=2, max_side=10),)
def test_gf_form_moments(args):
    """Check that the Gfs constructed from moments have the correct moment."""
    mom, = args
    gf = gt.fourier.pole_gf_from_moments(mom)
    gf_mom = gt.pole_gf_moments(poles=gf.poles, weights=gf.resids,
                                order=np.arange(mom.shape[-1])+1)
    assert np.allclose(mom, gf_mom, equal_nan=True)


def test_gf_form_moments_nan():
    """Check that the Gfs constructed from moments handle NaN."""
    mom = [np.nan]
    gf = gt.fourier.pole_gf_from_moments(mom)
    gf_mom = gt.pole_gf_moments(poles=gf.poles, weights=gf.resids, order=1)
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


# TODO: check if there is a way to improve result for pole↘0
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


@given(pole=st.floats(allow_nan=False, allow_infinity=False))
def test_tau2iw_dft_single_pole(pole):
    """Low accuracy test of `tau2iw_dft` on a single pole."""
    BETA = 1.3
    N_TAU = 8192 + 1
    tau = np.linspace(0, BETA, num=N_TAU)
    iws = gt.matsubara_frequencies(range(N_TAU//2), beta=BETA)

    gf_tau = gt.pole_gf_tau(tau, poles=[pole], weights=[1.], beta=BETA)
    gf_ft_lin = gt.fourier.tau2iw_dft(gf_tau + .5, beta=BETA) + 1/iws
    gf_iw = gt.pole_gf_z(iws, poles=[pole], weights=[1.])

    assert np.allclose(gf_iw, gf_ft_lin, atol=1e-4)
