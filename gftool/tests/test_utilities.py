# coding: utf8
"""Test the utility functions related to Green's functions."""
from __future__ import absolute_import, unicode_literals

from functools import partial

import pytest
import hypothesis.strategies as st

from hypothesis import given, assume

import numpy as np
import mpmath
from mpmath import fp

from .context import gftool as gt

approx = partial(np.allclose, rtol=1e-12, atol=1e-16, equal_nan=True)


def test_bose_edge_cases():
    assert gt.bose_fct(0., 1) == np.infty
    assert gt.bose_fct(np.infty, 1) == 0


@given(z=st.floats(min_value=1e-4, max_value=1e4), n=st.integers(min_value=-100, max_value=100))
@pytest.mark.parametrize("beta", [0.7, 1.38, 1000])
def test_complex_bose(z, n, beta):
    """Check if Bose function has imaginary periodicity.

    For bosonic Matsubara frequencies :math:`f(z+iν_n) = f(z)` should hold.
    """
    iv_n = gt.matsubara_frequencies_b(n, beta=beta)
    bose_cmpx = gt.bose_fct(z+iv_n, beta=beta)
    bose_real = gt.bose_fct(z, beta=beta)
    assert np.allclose(bose_cmpx.real, bose_real)
    assert bose_cmpx.imag < 1e-6*max(1, bose_real)


@given(z=st.floats())
@pytest.mark.parametrize("beta", [0.7, 1.38, 1000])
def test_fermi(z, beta):
    """Check if Fermi functions agrees with the standard form."""
    fermi_comp = 1./(np.exp(beta*z) + 1)
    # assert approx(fermi_comp) == gt.fermi_fct(z, beta=beta)
    assert approx(gt.fermi_fct(z, beta=beta), fermi_comp)


@given(z=st.floats(), n=st.integers(min_value=-100, max_value=100))
@pytest.mark.parametrize("beta", [0.7, 1.38, 1000])
def test_complex_fermi(z, n, beta):
    """Check if Fermi function has imaginary periodicity.

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
    """Check that `gt.fermi_fct_inv` is indeed the inverse.

    The other direction does not work in general, due to rounding.
    """
    assert approx(gt.fermi_fct(gt.fermi_fct_inv(z, beta), beta), z)


@given(z=st.floats())
@pytest.mark.parametrize("beta", [0.7, 1.38, 1000])
def test_fermi_d1_std_form(z, beta):
    """Check if Fermi functions agrees with the standard form."""
    assume(z*beta < 500.)  # avoid overflows in naive implementation
    exp = np.exp(beta*z)
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
    assert np.allclose(fp.diff(partial(gt.fermi_fct, beta=1), z, method='quad', radius=dist/2),
                       gt.fermi_fct_d1(z, beta=1))


def test_density():
    """Check density for simple Green's functions."""
    beta = 10.1
    D = 1.2
    iw_array = gt.matsubara_frequencies(np.arange(int(2**14)), beta=beta)

    #
    # Bethe lattice Gf
    #
    bethe_half = gt.bethe_gf_z(iw_array, half_bandwidth=D)
    assert gt.density(bethe_half, potential=0, beta=beta)[0] == 0.5

    large_shift = 10000*D
    # Bethe lattice almost filled (depends on small temperature)
    bethe_full = gt.bethe_gf_z(iw_array+large_shift, half_bandwidth=D)
    assert gt.density(bethe_full, potential=large_shift, beta=beta)[0] == pytest.approx(1.0)
    # Bethe lattice almost empty (depends on small temperature)
    bethe_empty = gt.bethe_gf_z(iw_array-large_shift, half_bandwidth=D)
    assert gt.density(bethe_empty, potential=-large_shift, beta=beta)[0] \
        == pytest.approx(0.0, abs=1e-6)

    #
    # single site
    #
    assert gt.density(iw_array, potential=0, beta=beta)[0] == pytest.approx(0.5)
