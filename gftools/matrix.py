r"""Functions to work with Green's in matrix from.

In the limit of infinite coordination number the self-energy becomes local,
inverse Gfs take the simple form:

.. math::

    (G^{-1}(i\omega))_{ii} &= i\omega - \mu_i - t_{ii} - \Sigma_i(i\omega)

    (G^{-1}(i\omega))_{ij} &= t_{ij} \quad \text{for } i \neq j


.. _`scipy.linalg.eig_banded`:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eig_banded.html
"""
import numpy as np
import scipy.linalg as la


def decompose_gf_omega_symmetric(g_inv_band):
    r"""Decompose the inverse Gf into eigenvalues and eigenvectors.
    
    The similarity transformation for symmetric matrices is orthogonal.

    .. math::
        G^{-1} = O h O^T, \quad h = diag(\lambda(G))
    
    Parameters
    ----------
    g_inv_band: (2, N) ndarray(complex)
        matrix to be decomposed, needs to be given in banded form
        (see `scipy.linalg.eig_banded`_)

    """
    h, rv = la.eig_banded(g_inv_band)
    rv_inv = rv.T  # symmetric matrix's are orthogonal diagonalizable
    # assert np.allclose(rv.dot(rv_inv), np.identity(*h.shape))
    return rv_inv, h, rv


def decompose_gf_omega(g_inv):
    r"""Decompose the inverse Gf into eigenvalues and eigenvectors.
    
    The similarity transformation:

    .. math::
        G^{-1} = P h P^{-1}, \quad h = diag(\lambda(G))
    
    Parameters
    ----------
    g_inv_band: (N, N) ndarray(complex)
        matrix to be decomposed
    """
    h, rv = la.eig(g_inv)
    rv_inv = la.inv(rv)
    return rv_inv, h, rv


def construct_gf_omega(rv_inv, diag_inv, rv):
    r"""Construct Gf from decomposition of its inverse.
    
    .. math::
        G^{-1} = P h P^{-1} \Rightarrow G = P h^{-1} P^{-1}

    Parameters
    ----------
    rv_inv: (N, N) ndarray(complex)
        The inverse of the matrix of right eigenvectors (:math:`P^{-1}`)
    diag_inv: (N) array_like
        The eigenvalues (:math:`h`)
    rv: (N, N) ndarray(complex)
        The matrix of right eigenvectors (:math:`P`)
    """
    return rv.dot(np.diag(diag_inv)).dot(rv_inv)
