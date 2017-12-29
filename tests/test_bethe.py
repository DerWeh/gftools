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
