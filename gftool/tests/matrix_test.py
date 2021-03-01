"""Tests of functions for real-space Gf matrices for infinite coordination number."""
from functools import partial

import pytest
import numpy as np
import scipy.linalg as la
import hypothesis.strategies as st

from hypothesis import given, assume
from hypothesis_gufunc.gufunc import gufunc_args

from .context import gftool as gt

easy_complex = st.complex_numbers(min_magnitude=1e-2, max_magnitude=1e+2)


class TestDecompositionGeneral:
    """Tests for the function `gftool.matrix.decompose_gf`.

    Main use for the function is to invert Green's functions,
    so we mainly test for that purpose.
    """

    omega_mesh = np.mgrid[-2:2:4j, -2:2:4j]  # don't use omega == 0
    omega = np.ravel(omega_mesh[0] + 1j*omega_mesh[1])
    g0_loc_inv = (omega)**-1

    @pytest.mark.parametrize("size", [4, 9, 20])
    def test_inverse_eigenvectors_non_interacting(self, size):
        """Eigenvector matrices for similarity transformations must be its inverse."""
        t_nn = np.ones(size-1)
        g0_inv_full = np.zeros((size, size), dtype=complex)
        g0_inv_full[np.arange(size-1), np.arange(size-1)+1] = t_nn
        g0_inv_full[np.arange(size-1)+1, np.arange(size-1)] = t_nn
        for g0 in self.g0_loc_inv:
            g0_inv_full[1] = g0
            rv, h, rv_inv = gt.matrix.decompose_gf(g0_inv_full)
            assert np.allclose(rv.dot(rv_inv), np.identity(*h.shape))

    @pytest.mark.parametrize("size", [4, 9, 20])
    def test_inverse_non_interacting(self, size):
        r"""Decomposition we be used to calculate the inverse.

        .. math::
            G^{-1} = P^{-1} h P \Rightarrow P^{-1} h^{-1} P = G
        """
        t_nn = np.ones(size-1)
        g0_inv_full = np.zeros((size, size), dtype=complex)
        g0_inv_full[np.arange(size-1), np.arange(size-1)+1] = t_nn
        g0_inv_full[np.arange(size-1)+1, np.arange(size-1)] = t_nn
        for g0 in self.g0_loc_inv:
            g0_inv_full[np.arange(size), np.arange(size)] = g0
            rv, h, rv_inv = gt.matrix.decompose_gf(g0_inv_full)
            g0 = gt.matrix.construct_gf(rv_inv=rv_inv, diag_inv=h**-1, rv=rv)
            assert np.allclose(g0.dot(g0_inv_full), np.identity(size))
            assert np.allclose(g0, la.inv(g0_inv_full))
            g0_alt = gt.matrix.Decomposition(rv, h**-1, rv_inv).reconstruct(kind='full')
            assert np.allclose(g0, g0_alt)

    @pytest.mark.parametrize("size", [4, 9, 20])
    def test_eigsum_non_interacting(self, size):
        """Trace of the matrix must be trace of eigenvalues *h*.

        This is due to cyclic invariance under the trace.
        """
        t_nn = np.ones(size-1)
        g0_inv_full = np.zeros((size, size), dtype=complex)
        g0_inv_full[np.arange(size-1), np.arange(size-1)+1] = t_nn
        g0_inv_full[np.arange(size-1)+1, np.arange(size-1)] = t_nn
        for g0 in self.g0_loc_inv:
            g0_inv_full[np.arange(size), np.arange(size)] = g0
            _, h, _ = gt.matrix.decompose_gf(g0_inv_full)
            assert np.allclose(np.sum(h), np.trace(g0_inv_full))


@given(gufunc_args('(n,n)->(n,n)', dtype=np.complex_, elements=easy_complex,
                   max_dims_extra=2, max_side=4),)
def test_decomposition_reconsturction(args):
    """Check if the reconstruction using `gt.matrix.Decomposition` is correct."""
    mat, = args  # unpack
    if mat.shape[-1] > 0:  # make sure matrix is diagonalizable
        assume(np.all(np.linalg.cond(mat) < 1e8))
    dec = gt.matrix.decompose_gf(mat)
    assert np.allclose(dec.reconstruct(kind='full'), mat)
    assert np.allclose(dec.reconstruct(kind='diag'), np.diagonal(mat, axis1=-2, axis2=-1))

    # Hermitian
    mat = mat + gt.matrix.transpose(mat).conj()
    if mat.shape[-1] > 0:  # make sure matrix is diagonalizable
        assume(np.all(np.linalg.cond(mat) < 1e8))
    dec = gt.matrix.decompose_hamiltonian(mat)
    assert np.allclose(dec.reconstruct(kind='full'), mat)
    assert np.allclose(dec.reconstruct(kind='diag'), np.diagonal(mat, axis1=-2, axis2=-1))


@given(gufunc_args('(n,n)->(n,n)', dtype=np.complex_, elements=easy_complex,
                   max_dims_extra=2, max_side=4),)
def test_decomposition_inverse(args):
    """Check if the inverse using `gt.matrix.Decomposition` is correct."""
    mat, = args  # unpack
    # make sure `mat` is reasonable
    if mat.shape[-1] > 0:  # make sure matrix is diagonalizable
        assume(np.all(np.linalg.cond(mat) < 1e8))
    inverse = np.linalg.inv(mat)
    dec = gt.matrix.Decomposition.from_gf(mat)
    assert np.allclose(dec.reconstruct(1./dec.xi, kind='full'), inverse)
    assert np.allclose(dec.reconstruct(1./dec.xi, kind='diag'),
                       np.diagonal(inverse, axis1=-2, axis2=-1))

    # Hermitian
    mat = mat + gt.matrix.transpose(mat).conj()
    if mat.shape[-1] > 0:  # make sure matrix is diagonalizable
        assume(np.all(np.linalg.cond(mat) < 1e8))
    inverse = np.linalg.inv(mat)
    dec = gt.matrix.Decomposition.from_hamiltonian(mat)
    assert np.allclose(dec.reconstruct(1./dec.xi, kind='full'), inverse)
    assert np.allclose(dec.reconstruct(1./dec.xi, kind='diag'),
                       np.diagonal(inverse, axis1=-2, axis2=-1))


@pytest.mark.filterwarnings("ignore:(overflow):RuntimeWarning")
@given(hopping=st.floats(min_value=-1e6, max_value=1e6),
       eps1=st.floats(min_value=-1e6, max_value=1e6),
       eps0=st.floats(min_value=-1e6, max_value=1e6),
       z=st.complex_numbers(allow_nan=False, allow_infinity=False))
def test_2x2_gf(z, eps0, eps1, hopping):
    """Compare analytic 2x2 Gf vs numeric diagonalization."""
    assume(abs(z.imag) > 1e-6)
    assume((eps0 != eps1) or (hopping != 0))
    ham = np.array([[eps0, hopping],
                    [hopping, eps1]])
    dec = gt.matrix.decompose_hamiltonian(ham)
    gf_num = dec.reconstruct(1/(z - dec.xi), kind='diag')
    assert np.allclose(gt.matrix.gf_2x2_z(z, eps0=eps0, eps1=eps1, hopping=hopping), gf_num)
    g0 = partial(gt.bethe_hilbert_transform, half_bandwidth=1)
    gf_num = dec.reconstruct(g0(z - dec.xi), kind='diag')
    gf_2x2 = gt.matrix.gf_2x2_z(z, eps0=eps0, eps1=eps1, hopping=hopping, hilbert_trafo=g0)
    assert np.allclose(gf_2x2, gf_num)
