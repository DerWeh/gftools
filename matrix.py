"""Functions to work with Green's in matrix from.

In the limit of infinite coordination number the self-energy becomes local,
inverse Gfs take the simple form:

.. math::

    (G^{-1}(i\omega))_{ii} = \iw - mu_i - t_{ii} - \Sigma_i(i\omega)
    (G^{-1}(i\omgea))_{ij} = t_{ij} \quad \text{for} i \neq j

"""
import numpy as np
import scipy.linalg as la


def decompose_gf_omega_symmetric(g_inv_band):
    """Decompose the inverse Gf into eigenvalues and eigenvectors.
    
    The similarity transformation for symmetric matrices is orthogonal.
    .. math::
        G^{-1} = O h O^T, \quad h = diag(\lambda(G))
    
    Params
    ------
    g_inv_band: (2, N) ndarray(complex)
        matrix to be decomposed, needs to be given in banded form
        (see *scipy.linalg.eig_banded*)
    """
    h, rv = la.eig_banded(g_inv_band)
    rv_inv = rv.T  # symmetric matrix's are orthogonal diagonalizable
    # assert np.allclose(rv.dot(rv_inv), np.identity(*h.shape))
    return rv_inv, h ,rv
