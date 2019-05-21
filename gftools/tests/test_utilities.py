# coding: utf8
"""Test the utility functions related to Green's functions."""
from __future__ import absolute_import, unicode_literals

from functools import partial

import pytest
import numpy as np
import hypothesis.strategies as st

from hypothesis import given, assume

from .context import gftools as gt

approx = partial(np.allclose, rtol=1e-12, atol=1e-16, equal_nan=True)


@given(z=st.floats())
@pytest.mark.parametrize("beta", [0.7, 1.38, 1000])
def test_fermi(z, beta):
    """Check if Fermi functions agrees with the standard form."""
    fermi_comp = 1./(np.exp(beta*z) + 1)
    # assert approx(fermi_comp) == gt.fermi_fct(z, beta=beta)
    assert approx(gt.fermi_fct(z, beta=beta), fermi_comp)


@given(z=st.floats())
@pytest.mark.parametrize("beta", [0.7, 1.38, 1000])
def test_fermi_d1(z, beta):
    """Check if Fermi functions agrees with the standard form."""
    assume(z*beta < 500.)  # avoid overflows in naive implementation
    exp = np.exp(beta*z)
    fermi_d1_comp = -beta*exp/(exp+1)**2
    assert approx(gt.fermi_fct_d1(z, beta=beta), fermi_d1_comp)


def test_density():
    """Check density for simple Green's functions."""
    beta = 10.1
    D = 1.2
    iw_array = gt.matsubara_frequencies(np.arange(int(2**14)), beta=beta)

    #
    # Bethe lattice Gf
    #
    bethe_half = gt.bethe_gf_omega(iw_array, half_bandwidth=D)
    assert gt.density(bethe_half, potential=0, beta=beta)[0] == 0.5

    large_shift = 10000*D
    # Bethe lattice almost filled (depends on small temperature)
    bethe_full = gt.bethe_gf_omega(iw_array+large_shift, half_bandwidth=D)
    assert gt.density(bethe_full, potential=large_shift, beta=beta)[0] == pytest.approx(1.0)
    # Bethe lattice almost empty (depends on small temperature)
    bethe_empty = gt.bethe_gf_omega(iw_array-large_shift, half_bandwidth=D)
    assert gt.density(bethe_empty, potential=-large_shift, beta=beta)[0] \
        == pytest.approx(0.0, abs=1e-6)

    #
    # single site
    #
    assert gt.density(iw_array, potential=0, beta=beta)[0] == pytest.approx(0.5)
