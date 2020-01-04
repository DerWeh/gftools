# coding: utf8
"""Test the Fourier transformation of Green's functions."""
import numpy as np
import hypothesis.strategies as st

from hypothesis import given

from .context import gftools as gt


@given(pole=st.floats(-10, 10))  # necessary Matsubaras depend on Bandwidth!
def test_iw2tau_dft_soft_single_pole(pole):
    """Low accurcy test of `iw2tau_dft_soft` on a single pole."""
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
    """Low accurcy test of `iw2tau_dft` on a single pole."""
    BETA = 1.3
    N_IW = 4096
    iws = gt.matsubara_frequencies(range(N_IW), beta=BETA)
    tau = np.linspace(0, BETA, num=2*N_IW+1)

    gf_iw = gt.pole_gf_z(iws, poles=[pole], weights=[1.])
    gf_dft = gt.fourier.iw2tau_dft(gf_iw - 1./iws, beta=BETA) - .5
    gf_tau = gt.pole_gf_tau(tau, poles=[pole], weights=[1.], beta=BETA)

    assert np.allclose(gf_tau, gf_dft, atol=1e-3, rtol=1e-4)


@given(pole=st.floats(allow_nan=False, allow_infinity=False))
def test_tau2iw_ft_lin_single_pole(pole):
    """Low accurcy test of `tau2iw_ft_lin` on a single pole."""
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
    """Low accurcy test of `tau2iw_dft` on a single pole."""
    BETA = 1.3
    N_TAU = 8192 + 1
    tau = np.linspace(0, BETA, num=N_TAU)
    iws = gt.matsubara_frequencies(range(N_TAU//2), beta=BETA)

    gf_tau = gt.pole_gf_tau(tau, poles=[pole], weights=[1.], beta=BETA)
    gf_ft_lin = gt.fourier.tau2iw_dft(gf_tau + .5, beta=BETA) + 1/iws
    gf_iw = gt.pole_gf_z(iws, poles=[pole], weights=[1.])

    assert np.allclose(gf_iw, gf_ft_lin, atol=1e-4)
