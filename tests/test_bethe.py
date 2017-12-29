"""Tests for functions related to Bethe GFs."""
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
    D = 1.
    for eps in np.linspace(D + 1e-6, D*1e4):
        assert gftools.bethe_dos(eps, D) == 0
        assert gftools.bethe_dos(-eps, D) == 0


def test_imag_gf_negative():
    """Imaginary part of Gf must be smaller or equal 0 for real frequencies."""
    D = 1.
    omega, omega_step = np.linspace(-D, D, dtype=np.complex, retstep=True)
    omega += 5j*omega_step
    assert np.all(gftools.bethe_gf_omega(omega, D).imag <= 0)


def test_imag_gf_equals_dos():
    r"""Imaginary part of the GF is proportional to the DOS.
    
    ..math: DOS(\epsilon) = - \Im(G(\epsilon))/\pi
    """
    D = 1.
    num=1e6
    omega, omega_step = np.linspace(-D, D, dtype=np.complex, retstep=True, num=num)
    omega += 5j*omega_step
    assert np.allclose(-gftools.bethe_gf_omega(omega, D).imag/np.pi,
                       gftools.bethe_dos(omega, D),
                       rtol=10/num, atol=10/num)
