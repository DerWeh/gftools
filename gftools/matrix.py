# encoding: utf-8
r"""Functions to work with Green's functions in matrix from.

In the limit of infinite coordination number the self-energy becomes local,
inverse Green's functions take the simple form:

.. math::

    (G^{-1}(iω))_{ii} &= iω - μ_i - t_{ii} - Σ_i(iω)

    (G^{-1}(iω))_{ij} &= t_{ij} \quad \text{for } i ≠ j

"""
from collections.abc import Sequence
from functools import partial

import numpy as np


transpose = partial(np.swapaxes, axis1=-2, axis2=-1)


class Decomposition(Sequence):
    """Decomposition of a Matrix into eigenvalues and eigenvectors.

    This class holds the eigenvalues and eigenvectors of the decomposition of a
    matrix and offers methods to reconstruct it.
    The intended use case is to use the `Decomposition` for the inversion of
    the Green's function to calculate it from the resolvent.

    The order of the attributes is always `rv, xi, rv_inv`, as this gives the
    reconstruct of the matrix:  `mat = (rv * xi) @ rv_inv`

    Attributes
    ----------
    rv : (..., N, N) complex np.ndarray
        The matrix of right eigenvalues.
    xi : (..., N) complex np.ndarray
        The vector of eigenvalues.
    rv_inv : (..., N, N) complex np.ndarray
        The inverse of `rv`.

    """

    __slots__ = ('rv', 'xi', 'rv_inv')

    def __init__(self, rv, xi, rv_inv):
        """Assign the attributes obtained from a matrix digitalization.

        Parameters
        ----------
        rv : (..., N, N) complex np.ndarray
            The matrix of right eigenvectors.
        xi : (..., N) complex np.ndarray
            The vector of eigenvalues
        rv_inv : (..., N, N) complex np.ndarray
            The inverse of `rv`.

        """
        self.rv = rv
        self.xi = xi
        self.rv_inv = rv_inv

    @classmethod
    def from_hamiltonian(cls, hamilton):
        r"""Decompose the Hamiltonian matrix.

        The similarity transformation:

        .. math:: H = U h U^†, \quad h = diag(λ_l)

        Parameters
        ----------
        hamilton : (..., N, N) complex np.ndarray
            Hermitian matrix to be decomposed

        Returns
        -------
        Decomposition

        """
        if isinstance(hamilton, cls):
            return hamilton
        return decompose_hamiltonian(hamilton)

    @classmethod
    def from_gf(cls, gf):
        r"""Decompose the inverse Green's function matrix.

        The similarity transformation:

        .. math:: G^{-1} = P g P^{-1}, \quad g = diag(λ_l)

        Parameters
        ----------
        g_inv : (..., N, N) complex np.ndarray
            matrix to be decomposed

        Returns
        -------
        Decomposition

        """
        if isinstance(gf, cls):
            return gf
        return decompose_gf(gf)

    def reconstruct(self, xi=None, kind='full'):
        """Get matrix back from `Decomposition`.

        If the reciprocal of `self.xi` was taken, this corresponds to the
        inverse of the original matrix.

        Parameters
        ----------
        xi : (..., N) np.ndarray, optional
            Alternative value used for `self.xi`. This argument can be used
            instead of modifying `self.xi`.
        kind : {'diag', 'full'} or str
            Defines how to reconstruct the matrix. If `kind` is 'diag',
            only the diagonal elements are computed, if it is 'full' the
            complete matrix is returned.
            Alternatively a `str` used for subscript of `np.einsum` can be given.

        Returns
        -------
        reconstruct : (..., N, N) or (..., N) np.ndarray
            The reconstructed matrix. If a subscript string is given as `kind`,
            the shape of the output might differ.

        """
        xi = xi if xi is not None else self.xi
        kind = kind.lower()
        if 'diag'.startswith(kind):
            return ((transpose(self.rv_inv)*self.rv) @ xi[..., np.newaxis])[..., 0]
        if 'full'.startswith(kind):
            return (self.rv * xi[..., np.newaxis, :]) @ self.rv_inv
        return np.einsum(kind, self.rv, xi, self.rv_inv)

    def __getitem__(self, key):
        """Make `Decomposition` behave like the tuple `(rv, xi, rv_inv)`."""
        return (self.rv, self.xi, self.rv_inv)[key]

    def __len__(self):
        return 3

    def __str__(self):
        return f"Decomposition({self.rv.shape}x{self.xi.shape}x{self.rv_inv.shape})"


def decompose_gf(g_inv) -> Decomposition:
    r"""Decompose the inverse Green's function into eigenvalues and eigenvectors.

    The similarity transformation:

    .. math:: G^{-1} = P g P^{-1}, \quad g = diag(λ_l)

    Parameters
    ----------
    g_inv : (..., N, N) complex np.ndarray
        matrix to be decomposed

    Returns
    -------
    Decomposition.rv : (..., N, N) complex np.ndarray
        The right eigenvectors :math:`P`
    Decomposition.h : (..., N) complex np.ndarray
        The complex eigenvalues of `g_inv`
    Decomposition.rv_inv : (..., N, N) complex np.ndarray
        The *inverse* of the right eigenvectors :math:`P`

    """
    if isinstance(g_inv, Decomposition):
        return g_inv
    h, rv = np.linalg.eig(g_inv)
    return Decomposition(rv=rv, xi=h, rv_inv=np.linalg.inv(rv))


def decompose_hamiltonian(hamilton) -> Decomposition:
    r"""Decompose the Hamiltonian matrix into eigenvalues and eigenvectors.

    The similarity transformation:

    .. math:: H = U h U^†, \quad h = diag(λ_l)

    Parameters
    ----------
    hamilton : (..., N, N) complex np.ndarray
        Hermitian matrix to be decomposed

    Returns
    -------
    Decomposition.rv : (..., N, N) complex np.ndarray
        The right eigenvectors :math:`U`
    Decomposition.h : (..., N) float np.ndarray
        The eigenvalues of `hamilton`
    Decomposition.rv_inv : (..., N, N) complex np.ndarray
        The *inverse* of the right eigenvectors :math:`U^†`. The Hamiltonian is
        hermitian, thus the decomposition is unitary :math:`U^† = U ^{-1}`

    """
    if isinstance(hamilton, Decomposition):
        return hamilton
    h, rv = np.linalg.eigh(hamilton)
    return Decomposition(rv=rv, xi=h, rv_inv=np.swapaxes(rv.conj(), -2, -1))


def construct_gf(rv, diag_inv, rv_inv):
    r"""Construct Green's function from decomposition of its inverse.

    .. math::
        G^{−1} = P h P^{-1} ⇒ G = P h^{-1} P^{-1}

    Parameters
    ----------
    rv_inv : (N, N) complex np.ndarray
        The inverse of the matrix of right eigenvectors (:math:`P^{-1}`)
    diag_inv : (N) array_like
        The eigenvalues (:math:`h`)
    rv : (N, N) complex np.ndarray
        The matrix of right eigenvectors (:math:`P`)

    Returns
    -------
    gf : (N, N) complex np.ndarray
        The Green's function

    """
    return rv.dot(np.diagflat(diag_inv)).dot(rv_inv)
