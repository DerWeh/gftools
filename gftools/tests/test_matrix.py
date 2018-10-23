"""Tests of functions for real-space Gf matrices for infinite coordination number."""
from __future__ import absolute_import

import pytest
import scipy.linalg as la

import numpy as np

from .context import gfmatrix


class TestDecompositionGeneral(object):
    """Tests for the function `gftools.matrix.decompose_gf_omega`.
    
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
            rv, h, rv_inv = gfmatrix.decompose_gf_omega(g0_inv_full)
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
            rv, h, rv_inv = gfmatrix.decompose_gf_omega(g0_inv_full)
            g0 = gfmatrix.construct_gf_omega(rv_inv=rv_inv, diag_inv=h**-1, rv=rv)
            assert np.allclose(g0.dot(g0_inv_full), np.identity(size))
            assert np.allclose(g0, la.inv(g0_inv_full))
            g0_alt = gfmatrix.Decomposition(rv, h**-1, rv_inv).reconstruct(kind='full')
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
            _, h, _ = gfmatrix.decompose_gf_omega(g0_inv_full)
            assert np.allclose(np.sum(h), np.trace(g0_inv_full))
