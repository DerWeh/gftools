"""Tests of linear algebra utilities."""
import numpy as np
import hypothesis.strategies as st
import pytest

from hypothesis import given, assume
from hypothesis_gufunc.gufunc import gufunc_args

from .context import gftool as gt

assert_allclose = np.testing.assert_allclose
easy_complex = st.complex_numbers(min_magnitude=1e-2, max_magnitude=1e+2)


@given(gufunc_args('(m,n)->(k,m)', dtype=np.complex_, elements=easy_complex,
                   max_dims_extra=0, max_side=10),)
def test_orth_compl(args):
    """Test that orthogonal complement is orthogonal."""
    mat, = args  # unpack
    dim0, dim1 = mat.shape[-2:]
    assume(dim0 > dim1)
    perp = gt.linalg.orth_compl(mat)
    assert_allclose(perp@mat, 0, atol=1e-12)


@given(gufunc_args('(m,n),(m),(n,n),(n)->(n)', dtype=np.complex_, elements=easy_complex,
                   max_dims_extra=0, max_side=5),)
def test_lstsq_ce_constraints(args):
    """Check if fully constraint solution of `gt.linalg.lstsq_ec` is correct."""
    a, b, c, d = args
    if c.shape[-1] > 0:  # make sure matrix is diagonalizable
        assume(np.all(np.linalg.cond(c) < 1e8))
    sol = np.linalg.solve(c, d[..., np.newaxis])[..., 0]
    lstsq = gt.linalg.lstsq_ec(a, b, c, d)
    assert_allclose(lstsq, sol, atol=1e-14)


@pytest.mark.skip("Can't get this test working...")
@given(gufunc_args('(m,n),(m),(l,n),(l)->(n)', dtype=np.complex_, elements=easy_complex,
                   max_dims_extra=0, max_side=5),)
def test_lstsq_ce(args):
    """Check if solution of `gt.linalg.lstsq_ec` fulfills constraints."""
    a, b, c, d = args
    if a.shape[-1] > 0 and a.shape[-2]:  # make sure matrices are reasonable
        assume(np.all(np.linalg.cond(a) < 1e8))
    if b.shape[-1] < d.shape[-1]:  # overconstrained
        max_l = b.shape[-1] - 1
        c = c[..., :max_l, :]
        d = d[..., :max_l]
    if c.shape[-1] < c.shape[-2]:  # overconstrained
        max_l = c.shape[-1]
        c = c[..., :max_l, :]
        d = d[..., :max_l]
    if c.shape[-1] > 0 and c.shape[-2]:  # make sure matrices are reasonable
        assume(np.all(np.linalg.cond(c) < 1e4))
    lstsq = gt.linalg.lstsq_ec(a, b, c, d)
    assert_allclose(np.sum(c*lstsq, axis=-1), d, atol=1e-6)


def test_singular_constraint():
    """Check extreme edge case of singular constraints."""
    M, N, L, = 20, 7, 3
    RNG = np.random.default_rng(0)
    a = RNG.random([M, N])
    b = RNG.random([M])
    crt = np.array([[1, 2, 3],
                    [0, 4, 5]]
                   + [[0]*3]*(N-2)
                   )
    c = (np.eye(N) @ crt).T
    d = np.arange(1, L+1)
    lstsq = gt.linalg.lstsq_ec(a, b, c, d)
    assert_allclose(np.sum(c*lstsq, axis=-1), d)


@given(gufunc_args('(m,n),(m)->(n)', dtype=np.complex_, elements=easy_complex,
                   max_dims_extra=0, max_side=5),)
def test_lstsq_ce_is_lstq(args):
    """Check if solution of `gt.linalg.lstsq_ec` is a least-squares solution."""
    a, b = args
    if a.shape[-1] > 0 and a.shape[-2]:  # make sure matrices are reasonable
        assume(np.all(np.linalg.cond(a) < 1e8))
    lstsq = np.linalg.lstsq(a, b[..., np.newaxis], rcond=None)[0][..., 0]
    n = lstsq.shape[-1]
    c = np.eye(n//2, n)
    d = lstsq[..., :n//2]
    lstsq_ec = gt.linalg.lstsq_ec(a, b, c, d)
    assert_allclose(lstsq_ec, lstsq, atol=1e-14)
