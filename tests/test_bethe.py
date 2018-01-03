"""Tests for functions related to Bethe GFs.

TODO: use accuracy of *integrate.quad* for *pytest.approx*
"""
from __future__ import absolute_import

import pytest
import scipy.integrate as integrate

import numpy as np

from .context import gftools


@pytest.mark.parametrize("D", [0.5, 1., 2.])
def test_dos_unit(D):
    """Integral over the whole DOS should be 1."""
    assert integrate.quad(gftools.bethe_dos, -D-.1, D+.1, args=(D,))[0] == pytest.approx(1.)


@pytest.mark.parametrize("D", [0.5, 1., 2.])
def test_dos_half(D):
    """DOS should be symmetric -> integral over the half should yield 0.5."""
    assert integrate.quad(gftools.bethe_dos, -D-.1, 0., args=(D,))[0] == pytest.approx(.5)
    assert integrate.quad(gftools.bethe_dos, 0., D+.1, args=(D,))[0] == pytest.approx(.5)


def test_dos_support():
    """DOS should have no support for |eps| > D."""
    D = 1.2
    for eps in np.linspace(D + 1e-6, D*1e4):
        assert gftools.bethe_dos(eps, D) == 0
        assert gftools.bethe_dos(-eps, D) == 0


def test_imag_gf_negative():
    """Imaginary part of Gf must be smaller or equal 0 for real frequencies."""
    D = 1.2
    omega, omega_step = np.linspace(-D, D, dtype=np.complex, retstep=True)
    omega += 5j*omega_step
    assert np.all(gftools.bethe_gf_omega(omega, D).imag <= 0)


@pytest.mark.skip(reason="Accuracy around bandeges is to low!")
def test_imag_gf_equals_dos():
    r"""Imaginary part of the GF is proportional to the DOS.
    
    .. math::
        DOS(\epsilon) = -\Im(G(\epsilon))/\pi
    """
    D = 1.2
    num=1e6
    omega = np.linspace(-D, D, dtype=np.complex, num=num)
    omega += 1j*1e-16
    assert np.allclose(-gftools.bethe_gf_omega(omega, D).imag/np.pi,
                       gftools.bethe_dos(omega, D))


def test_hilbert_equals_integral():
    """Compare *bethe_hilbert_transfrom* with explicit calculation of integral.
    
    The integral is singular for xi=0, actually the Cauchy principal value
    should be taken.
    """
    D = 1.
    xi_mesh = np.mgrid[-2*D:2*D:4j, -2*D:2*D:4j]
    xi_values = np.ravel(xi_mesh[0] + 1j*xi_mesh[1])

    def kernel(eps, xi):
        """Integrand for the Hilbert transform"""
        return gftools.bethe_dos(eps, half_bandwidth=D)/(xi - eps)

    for xi in xi_values:
        compare = 0
        compare += integrate.quad(lambda eps: kernel(eps, xi).real, -D-.1, D+.1)[0]
        compare += 1j*integrate.quad(lambda eps: kernel(eps, xi).imag, -D-.1, D+.1)[0]
        assert gftools.bethe_hilbert_transfrom(xi, D) == pytest.approx(compare)
