"""Test Padé against `scipy` implementation."""
import hypothesis.strategies as st
import numpy as np
import pytest

from hypothesis import given
from scipy import interpolate
from scipy.special import binom

from .context import gftool as gt

ignore_illconditioned = pytest.mark.filterwarnings(
    "ignore:(Ill-conditioned matrix):scipy.linalg.LinAlgWarning"
)


@ignore_illconditioned
@given(num_deg=st.integers(0, 10), den_deg=st.integers(0, 10))
def test_pade_vs_scipy(num_deg, den_deg):
    """Compare against the more naive `scipy` algorithm.

    For small degrees they should be equal.
    """
    deg = num_deg + den_deg + 1
    an = np.arange(1, deg+1)
    pade = gt.hermpade.pade(an, num_deg=num_deg, den_deg=den_deg)
    try:
        pade_sp = interpolate.pade(an, m=den_deg, n=num_deg)
    except np.linalg.LinAlgError:  # scipy cannot handle this
        return

    test_values = np.array([0, 1, np.pi, np.sqrt(2), np.e])
    for val in test_values:
        if pade.denom(val) < 1e-8 or pade_sp[1](val) < 1e-8:
            continue  # near singular
        assert np.allclose(pade.eval(val),
                           pade_sp[0](val)/pade_sp[1](val))
    # # scipy uses q[0] = 1, while we enforce nothing
    # factor = pade.denom.coef[0]
    # # note that scipy returns outdated `poly1d` with reverse order
    # # it also truncates trailing zeros
    # assert np.allclose(pade.numer.coef, factor*pade_sp[0].coef[::-1])
    # assert np.allclose(pade.denom.coef, factor*pade_sp[1].coef[::-1])


@ignore_illconditioned
@given(num_deg=st.integers(0, 10), den_deg=st.integers(0, 10))
@pytest.mark.parametrize("fast", [True, False])
def test_pade_for_cubic(num_deg, den_deg, fast):
    """Compare for cubic root function, where there are no issues for Padé."""
    deg = num_deg + den_deg + 1
    an = binom(1/3, np.arange(deg))  # Taylor of (1+x)**(1/3)
    pade = gt.hermpade.pade(an, num_deg=num_deg, den_deg=den_deg, fast=fast)
    pade_sp = interpolate.pade(an, m=den_deg, n=num_deg)

    test_values = np.array([0, 1, np.pi, np.sqrt(2), np.e])
    assert np.allclose(pade.eval(test_values), pade_sp[0](test_values)/pade_sp[1](test_values))


def test_cubic_root():
    """Test approximants against cubic root function."""
    an = binom(1/3, np.arange(17))  # Taylor of (1+x)**(1/3)
    x = np.linspace(-0.5, 2, num=500)
    fx = np.emath.power(1+x, 1/3)
    pade = gt.hermpade.pade(an, den_deg=8, num_deg=8)
    assert np.allclose(pade.eval(x), fx, rtol=1e-8)
    herm = gt.hermpade.Hermite2.from_taylor(an, 5, 5, 5)
    assert np.allclose(herm.eval(x), fx, rtol=1e-10)


def test_single_pole():
    """Test approximants against a single pole."""
    an = (-1)**np.arange(17)
    x = np.linspace(-3, 3, num=500)
    fx = 1 / (x + 1)
    pade = gt.hermpade.pade(an, den_deg=8, num_deg=8)
    assert np.allclose(pade.eval(x), fx, rtol=1e-14, atol=1e-12)
    herm = gt.hermpade.Hermite2.from_taylor(an, 5, 5, 5)
    assert np.allclose(herm.eval(x), fx, rtol=1e-11, atol=1e-12)


def test_square_root():
    """Square Hermite-Padé should be exact for the right branch."""
    an = binom(1/2, np.arange(17))  # Taylor of (1+x)**(1/3)
    x = np.linspace(-3, 3, num=500)
    fx = np.emath.sqrt(1+x)
    herm = gt.hermpade.Hermite2.from_taylor(an, 5, 5, 5)
    p_branch, __ = herm.eval_branches(x)
    assert np.allclose(p_branch, fx, rtol=1e-14)
