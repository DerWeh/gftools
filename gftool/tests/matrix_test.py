"""Tests of functions for real-space Gf matrices for infinite coordination number."""
from functools import partial

import hypothesis.strategies as st
import numpy as np
import pytest
import scipy.linalg as la
from hypothesis import assume, given

import gftool as gt
from gftool._precision import HAS_QUAD
from gftool.tests.custom_strategies import gufunc_args

assert_allclose = np.testing.assert_allclose
easy_complex = st.complex_numbers(min_magnitude=1e-2, max_magnitude=1e+2)


class TestDecompositionGeneral:
    """
    Tests for the function `gftool.matrix.decompose_gf`.

    Main use for the function is to invert Green's functions,
    so we mainly test for that purpose.
    """

    omega_mesh = np.mgrid[-2:2:4j, -2:2:4j]  # don't use omega == 0
    omega = np.ravel(omega_mesh[0] + 1j*omega_mesh[1])
    g0_loc_inv = (omega)**-1

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.parametrize("size", [4, 9, 20])
    def test_inverse_eigenvectors_non_interacting(self, size):
        """Eigenvector matrices for similarity transformations must be its inverse."""
        t_nn = 1.2
        idx = np.arange(size)
        g0_inv_full = np.zeros((size, size), dtype=complex)
        g0_inv_full[idx[:-1], idx[1:]] = g0_inv_full[idx[1:], idx[:-1]] = t_nn
        for g0 in self.g0_loc_inv:
            g0_inv_full[idx, idx] = g0
            rv, h, rv_inv = gt.matrix.decompose_gf(g0_inv_full)
            assert_allclose(rv.dot(rv_inv), np.identity(*h.shape), atol=1e-14)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.parametrize("size", [4, 9, 20])
    def test_inverse_non_interacting(self, size):
        r"""
        Decomposition we be used to calculate the inverse.

        .. math::
            G^{-1} = P^{-1} h P \Rightarrow P^{-1} h^{-1} P = G
        """
        t_nn = 1.2
        idx = np.arange(size)
        g0_inv_full = np.zeros((size, size), dtype=complex)
        g0_inv_full[idx[:-1], idx[1:]] = g0_inv_full[idx[1:], idx[:-1]] = t_nn
        for g0 in self.g0_loc_inv:
            g0_inv_full[idx, idx] = g0
            rv, h, rv_inv = gt.matrix.decompose_gf(g0_inv_full)
            g0mat = gt.matrix.construct_gf(rv_inv=rv_inv, diag_inv=h**-1, rv=rv)
            assert_allclose(g0mat.dot(g0_inv_full), np.identity(size), atol=1e-14)
            assert_allclose(g0mat, la.inv(g0_inv_full))
            g0_alt = gt.matrix.Decomposition(rv, h**-1, rv_inv).reconstruct(kind='full')
            assert_allclose(g0mat, g0_alt)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.parametrize("size", [4, 9, 20])
    def test_eigsum_non_interacting(self, size):
        """
        Trace of the matrix must be trace of eigenvalues *h*.

        This is due to cyclic invariance under the trace.
        """
        t_nn = 1.2
        idx = np.arange(size)
        g0_inv_full = np.zeros((size, size), dtype=complex)
        g0_inv_full[idx[:-1], idx[1:]] = g0_inv_full[idx[1:], idx[:-1]] = t_nn
        for g0 in self.g0_loc_inv:
            g0_inv_full[idx, idx] = g0
            _, h, _ = gt.matrix.decompose_gf(g0_inv_full)
            assert_allclose(np.sum(h), np.trace(g0_inv_full))


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@given(
    gufunc_args(
        shape_kwds={"signature": "(n,n)->(n,n)"},
        dtype=np.complex128,
        elements=easy_complex,
    )
)
def test_decomposition_reconsturction(guargs):
    """Check if the reconstruction using `gt.matrix.Decomposition` is correct."""
    (mat,) = guargs.args  # unpack
    if mat.shape[-1] > 0:  # make sure matrix is diagonalizable
        assume(np.all(np.linalg.cond(mat) < 1e8))
    dec = gt.matrix.decompose_gf(mat)
    assert_allclose(dec.reconstruct(kind='full'), mat, atol=1e-10)
    assert_allclose(dec.reconstruct(kind='diag'),
                    np.diagonal(mat, axis1=-2, axis2=-1), atol=1e-10)

    # symmetric
    sym_mat = mat + gt.matrix.transpose(mat)
    if sym_mat.shape[-1] > 0:  # make sure matrix is diagonalizable
        assume(np.all(np.linalg.cond(sym_mat) < 1e8))
    dec = gt.matrix.decompose_sym(sym_mat)
    assert_allclose(dec.reconstruct(kind='full'), sym_mat, atol=1e-10)
    assert_allclose(dec.reconstruct(kind='diag'),
                    np.diagonal(sym_mat, axis1=-2, axis2=-1), atol=1e-10)

    # Hermitian
    her_mat = mat + gt.matrix.transpose(mat).conj()
    if her_mat.shape[-1] > 0:  # make sure matrix is diagonalizable
        assume(np.all(np.linalg.cond(her_mat) < 1e8))
    dec = gt.matrix.decompose_hamiltonian(her_mat)
    assert_allclose(dec.reconstruct(kind='full'), her_mat, atol=1e-10)
    assert_allclose(dec.reconstruct(kind='diag'),
                    np.diagonal(her_mat, axis1=-2, axis2=-1), atol=1e-10)


@pytest.mark.filterwarnings("ignore:(overflow)|(invalid value):RuntimeWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@given(
    gufunc_args(
        shape_kwds={"signature": "(n,n)->(n,n)"},
        dtype=np.complex128,
        elements=easy_complex,
    )
)
def test_decomposition_inverse(guargs):
    """Check if the inverse using `gt.matrix.Decomposition` is correct."""
    (mat,) = guargs.args  # unpack
    # make sure `mat` is reasonable
    if mat.shape[-1] > 0:  # make sure matrix is diagonalizable
        assume(np.all(np.linalg.cond(mat) < 1e8))
    inverse = np.linalg.inv(mat)
    dec = gt.matrix.Decomposition.from_gf(mat)
    assert_allclose(dec.reconstruct(1./dec.eig, kind='full'), inverse, atol=1e-10)
    assert_allclose(dec.reconstruct(1./dec.eig, kind='diag'),
                    np.diagonal(inverse, axis1=-2, axis2=-1), atol=1e-10)

    # symmetric
    sym_mat = mat + gt.matrix.transpose(mat)
    if sym_mat.shape[-1] > 0:  # make sure matrix is diagonalizable
        assume(np.all(np.linalg.cond(sym_mat) < 1e8))
    inverse = np.linalg.inv(sym_mat)
    dec = gt.matrix.decompose_sym(sym_mat)
    assert_allclose(dec.reconstruct(1./dec.eig, kind='full'), inverse, atol=1e-10)
    assert_allclose(dec.reconstruct(1./dec.eig, kind='diag'),
                    np.diagonal(inverse, axis1=-2, axis2=-1), atol=1e-10)

    # Hermitian
    her_mat = mat + gt.matrix.transpose(mat).conj()
    if her_mat.shape[-1] > 0:  # make sure matrix is diagonalizable
        assume(np.all(np.linalg.cond(her_mat) < 1e8))
    inverse = np.linalg.inv(her_mat)
    dec = gt.matrix.Decomposition.from_hamiltonian(her_mat)
    assert_allclose(dec.reconstruct(1./dec.eig, kind='full'), inverse, atol=1e-10)
    assert_allclose(dec.reconstruct(1./dec.eig, kind='diag'),
                    np.diagonal(inverse, axis1=-2, axis2=-1), atol=1e-10)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore:(overflow):RuntimeWarning")
@given(hopping=st.floats(min_value=-1e6, max_value=1e6),
       eps1=st.floats(min_value=-1e6, max_value=1e6),
       eps0=st.floats(min_value=-1e6, max_value=1e6),
       z=st.complex_numbers(allow_nan=False, allow_infinity=False,
                            max_magnitude=None if HAS_QUAD else 1e64))
def test_2x2_gf(z, eps0, eps1, hopping):
    """Compare analytic 2x2 Gf vs numeric diagonalization."""
    assume(abs(z.imag) > 1e-6)
    assume(abs(eps0 - eps1) > 1e-16 or abs(hopping) > 1e-16)
    ham = np.array([[eps0, hopping],
                    [hopping, eps1]])
    dec = gt.matrix.decompose_hamiltonian(ham)
    gf_num = dec.reconstruct(1/(z - dec.eig), kind='diag')
    assert_allclose(gt.matrix.gf_2x2_z(z, eps0=eps0, eps1=eps1, hopping=hopping),
                    gf_num, rtol=1e-5, atol=1e-14)
    g0 = partial(gt.bethe_hilbert_transform, half_bandwidth=1)
    gf_num = dec.reconstruct(g0(z - dec.eig), kind='diag')
    gf_2x2 = gt.matrix.gf_2x2_z(z, eps0=eps0, eps1=eps1, hopping=hopping, hilbert_trafo=g0)
    assert_allclose(gf_2x2, gf_num, rtol=1e-5, atol=1e-14)
