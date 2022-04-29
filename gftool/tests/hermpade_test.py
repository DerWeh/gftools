"""Test Padé against `scipy` implementation."""
# pylint: disable=protected-access
from functools import partial

import hypothesis.strategies as st
import numpy as np
import pytest

from hypothesis import given
from scipy import interpolate
from scipy.special import binom

from .context import gftool as gt

assert_allclose = np.testing.assert_allclose
ignore_illconditioned = pytest.mark.filterwarnings(
    "ignore:(Ill-conditioned matrix):scipy.linalg.LinAlgWarning"
)


def test_nullvec_lstsq():
    """Compare null-vector least square against regular null-vector.

    As test-case we consider a typical Padé matrix.
    """
    size = 17
    amat = np.random.default_rng(0).normal(size=[size-1, size])
    qcoef = gt.hermpade._nullvec(amat)
    qcoef *= np.sign(qcoef[0])  # fix sign
    for fix in [0, 8, -7, -1]:
        qcoef_lst = gt.hermpade._nullvec_lst(amat, fix=fix, rcond=0)
        qcoef_lst *= np.sign(qcoef_lst[0])  # fix sign
        assert_allclose(qcoef_lst/np.linalg.norm(qcoef_lst), qcoef)


@ignore_illconditioned
@given(num_deg=st.integers(0, 10), den_deg=st.integers(0, 10))
@pytest.mark.parametrize("pade", [gt.hermpade.pade, gt.hermpade.pade_lstsq])
def test_pade_vs_scipy(num_deg, den_deg, pade):
    """Compare against the more naive `scipy` algorithm.

    For small degrees they should be equal.
    """
    deg = num_deg + den_deg + 1
    an = np.arange(1, deg+1)
    ratpol = pade(an, num_deg=num_deg, den_deg=den_deg)
    try:
        pade_sp = interpolate.pade(an, m=den_deg, n=num_deg)
    except np.linalg.LinAlgError:  # scipy cannot handle this
        return

    test_values = np.array([0, 1, np.pi, np.sqrt(2), np.e])
    for val in test_values:
        if ratpol.denom(val) < 1e-8 or pade_sp[1](val) < 1e-8:
            continue  # near singular
        assert_allclose(ratpol.eval(val), pade_sp[0](val)/pade_sp[1](val))
    # # scipy uses q[0] = 1, while we enforce nothing
    # factor = pade.denom.coef[0]
    # # note that scipy returns outdated `poly1d` with reverse order
    # # it also truncates trailing zeros
    # assert_allclose(pade.numer.coef, factor*pade_sp[0].coef[::-1])
    # assert_allclose(pade.denom.coef, factor*pade_sp[1].coef[::-1])


def test_strip_coeffs():
    """Check stripping of coefficients that are zero."""
    pc = np.array([0, 1, 0, 0, 0, 3, 0, 2])
    qc = np.array([0, 1, 2, 0, 0, 3, 0, 0, 0])
    pc, qc = gt.hermpade._strip_ceoffs(pc, qc)
    assert_allclose(pc, [1, 0, 0, 0, 3, 0, 2])
    assert_allclose(qc, [1, 2, 0, 0, 3])
    pc = np.array([1, 0, 3, 0, 2, 0])
    qc = np.array([1, 2, 0, 0, 3, 0, 0])
    pc, qc = gt.hermpade._strip_ceoffs(pc, qc)
    assert_allclose(pc, [1, 0, 3, 0, 2])
    assert_allclose(qc, [1, 2, 0, 0, 3])
    pc = np.array([1, 2, 3, 0])
    qc = np.array([1, 2, 3, 4])
    pc, qc = gt.hermpade._strip_ceoffs(pc, qc)
    assert_allclose(pc, [1, 2, 3])
    assert_allclose(qc, [1, 2, 3, 4])


def test_pader_degree():
    r"""Check if `pader` can recover the degree for simple example.

    We consider the function

    .. math::

       (x+3)/(x-1)(x+1) = (x+3)/(x^2 - 1) = \sum_{n=0}^{∞}[-2+{(-1)}^{1+n}]x^n

    """
    deg = 400
    ns = np.arange(deg)
    an = (-2 + (-1)**(1+ns))
    approx = gt.hermpade.pader(an, num_deg=deg//2-1, den_deg=deg//2)
    assert approx.numer.degree() == 1
    assert approx.denom.degree() == 2
    xx = np.linspace(-0.5, 0.5, num=100)
    assert_allclose(approx.eval(xx), (xx+3) / (xx**2 - 1))


@ignore_illconditioned
@given(num_deg=st.integers(0, 10), den_deg=st.integers(0, 10))
@pytest.mark.parametrize("pade", [gt.hermpade.pade,
                                  partial(gt.hermpade.pade, fast=True),
                                  gt.hermpade.pade_lstsq
                                  ])
def test_pade_for_cubic(num_deg, den_deg, pade):
    """Compare for cubic root function, where there are no issues for Padé."""
    deg = num_deg + den_deg + 1
    an = binom(1/3, np.arange(deg))  # Taylor of (1+x)**(1/3)
    ratpol = pade(an, num_deg=num_deg, den_deg=den_deg)
    pade_sp = interpolate.pade(an, m=den_deg, n=num_deg)

    test_values = np.array([0, 1, np.pi, np.sqrt(2), np.e])
    assert_allclose(ratpol.eval(test_values), pade_sp[0](test_values)/pade_sp[1](test_values))


def test_cubic_root():
    """Test approximants against cubic root function."""
    an = binom(1/3, np.arange(17))  # Taylor of (1+x)**(1/3)
    x = np.linspace(-0.5, 2, num=500)
    fx = np.emath.power(1+x, 1/3)
    pade = gt.hermpade.pade(an, den_deg=8, num_deg=8)
    assert_allclose(pade.eval(x), fx, rtol=1e-8)
    pade = gt.hermpade.pade_lstsq(an, den_deg=8, num_deg=8)
    assert_allclose(pade.eval(x), fx, rtol=1e-8)
    herm = gt.hermpade.Hermite2.from_taylor(an, 5, 5, 5)
    assert_allclose(herm.eval(x), fx, rtol=3e-10)
    herm = gt.hermpade.Hermite2.from_taylor_lstsq(an, 5, 5, 5)
    assert_allclose(herm.eval(x), fx, rtol=3e-10)


def test_single_pole():
    """Test approximants against a single pole."""
    an = (-1)**np.arange(17)
    x = np.linspace(-3, 3, num=500)
    fx = 1 / (x + 1)
    pade = gt.hermpade.pade(an, den_deg=8, num_deg=8)
    assert_allclose(pade.eval(x), fx, rtol=1e-14, atol=1e-12)
    pade = gt.hermpade.pade_lstsq(an, den_deg=8, num_deg=8)
    assert_allclose(pade.eval(x), fx, rtol=1e-12, atol=2e-12)
    herm = gt.hermpade.Hermite2.from_taylor(an, 5, 5, 5)
    assert_allclose(herm.eval(x), fx, rtol=1e-11, atol=1e-12)
    herm = gt.hermpade.Hermite2.from_taylor_lstsq(an, 5, 5, 5)
    assert_allclose(herm.eval(x), fx, rtol=1e-11, atol=1e-12)


def test_square_root():
    """Square Hermite-Padé should be exact for the right branch."""
    an = binom(1/2, np.arange(17))  # Taylor of (1+x)**(1/3)
    x = np.linspace(-3, 3, num=500)
    fx = np.emath.sqrt(1+x)
    herm = gt.hermpade.Hermite2.from_taylor(an, 5, 5, 5)
    p_branch, __ = herm.eval_branches(x)
    assert_allclose(p_branch, fx, rtol=1e-14)
    herm = gt.hermpade.Hermite2.from_taylor_lstsq(an, 5, 5, 5)
    p_branch, __ = herm.eval_branches(x)
    assert_allclose(p_branch, fx, rtol=1e-14)
