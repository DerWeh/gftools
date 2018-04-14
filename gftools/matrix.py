# encoding: utf-8
r"""Functions to work with Green's in matrix from.

In the limit of infinite coordination number the self-energy becomes local,
inverse Green's functions take the simple form:

.. math::

    (G^{-1}(iω))_{ii} &= iω - μ_i - t_{ii} - Σ_i(iω)

    (G^{-1}(iω))_{ij} &= t_{ij} \quad \text{for } i ≠ j

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import scipy.linalg as la


def decompose_gf_omega_symmetric(g_inv_band):
    r"""Decompose the Green's function into eigenvalues and eigenvectors.
    
    The similarity transformation for symmetric matrices is orthogonal.

    .. math::
        G^{-1} = O h O^T, \quad h = diag(λ(G))
    
    Parameters
    ----------
    g_inv_band : (2, N) ndarray(complex)
        matrix to be decomposed, needs to be given in banded form
        (see :func:`scipy.linalg.eig_banded`)

    Returns
    -------
    rv_inv : (N, N) ndarray(complex)
        The *inverse* of the right eigenvectors :math:`O`
    h : (N) ndarray(complex)
        The complex eigenvalues of `g_inv_band`
    rv : (N, N) ndarray(complex)
        The right eigenvectors :math:`O`

    """
    h, rv = la.eig_banded(g_inv_band)
    rv_inv = rv.T  # symmetric matrix's are orthogonal diagonalizable
    # assert np.allclose(rv.dot(rv_inv), np.identity(*h.shape))
    return rv_inv, h, rv


def decompose_gf_omega(g_inv):
    r"""Decompose the inverse Green's function into eigenvalues and eigenvectors.
    
    The similarity transformation:

    .. math::
        G^{-1} = P h P^{-1}, \quad h = diag(λ(G))
    
    Parameters
    ----------
    g_inv : (N, N) ndarray(complex)
        matrix to be decomposed

    Returns
    -------
    rv_inv : (N, N) ndarray(complex)
        The *inverse* of the right eigenvectors :math:`P`
    h : (N) ndarray(complex)
        The complex eigenvalues of `g_inv`
    rv : (N, N) ndarray(complex)
        The right eigenvectors :math:`P`

    """
    h, rv = la.eig(g_inv)
    rv_inv = la.inv(rv)
    return rv_inv, h, rv


def construct_gf_omega(rv_inv, diag_inv, rv):
    r"""Construct Green's function from decomposition of its inverse.
    
    .. math::
        G^{−1} = P h P^{-1} ⇒ G = P h^{-1} P^{-1}

    Parameters
    ----------
    rv_inv : (N, N) ndarray(complex)
        The inverse of the matrix of right eigenvectors (:math:`P^{-1}`)
    diag_inv : (N) array_like
        The eigenvalues (:math:`h`)
    rv : (N, N) ndarray(complex)
        The matrix of right eigenvectors (:math:`P`)

    Returns
    -------
    gf_omega : (N, N) ndarray(complex)
        The Green's function

    """
    return rv.dot(np.diag(diag_inv)).dot(rv_inv)
