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


class Decomposition(object):
    """Abstraction for matrix decomposition designed for green's functions.

    This can be used as drop-in replacement for returning `rv, xi, rv_inv`.
    Additionally it offers methods to reconstruct that matrix.
    The intended use case is to use the `Decomposition` for the inversion of
    the Green's function to calculate it from the resolvent.

    Attributes
    ----------
    rv : (N, N) complex np.ndarray
        The matrix of right eigenvalues.
    xi : (N, ...) complex np.ndarray
        The vector of eigenvalues
    rv_inv : (N, N) complex np.ndarray
        The inverse of `rv`.

    """

    __slots__ = ('rv', 'xi', 'rv_inv')

    def __init__(self, rv, xi, rv_inv):
        """Assign the attributes obtained from a matrix digitalization.

        Parameters
        ----------
        rv : (N, N) complex np.ndarray
            The matrix of right eigenvalues.
        xi : (N, ...) complex np.ndarray
            The vector of eigenvalues
        rv_inv : (N, N) complex np.ndarray
            The inverse of `rv`.

        """
        self.rv = rv
        self.xi = xi
        self.rv_inv = rv_inv

    def reconstruct(self, xi=None, kind='full'):
        """Get matrix back from `Decomposition`.

        If the reciprocal of `self.xi` was taken, this corresponds to the
        inverse of the original matrix.
        New axises can be appended to `self.xi`, however the 0th axis has to
        correspond to the matrix dimension.

        Parameters
        ----------
        xi : (N, ...) ndarray, optional
            Alternative value used for `self.xi`. This argument can be used
            instead of calling `self.apply` or modifying `self.xi`.
        kind : {'diag', 'full'} or str
            Defines how to reconstruct the matrix. If `kind` is 'diag',
            only the diagonal elements are returned, if it is 'full' the
            complete matrix is returned. The start of the strings is also
            sufficient (e.g. `kind` = 'f').
            Alternatively as `str` used for `np.einsum` can be given.

        Returns
        -------
        reconstruct : (N, N, ...) or (N, ...) ndarray
            The reconstructed matrix.

        """
        xi = xi if xi is not None else self.xi
        kind = kind.lower()
        if 'diag'.startswith(kind):
            return self._reconstruct_diag(xi)
        if 'full'.startswith(kind):
            return self._reconstruct_full(xi)
        return self._reconstruct_einsum(xi, sum_str=kind)

    def _reconstruct_diag(self, xi):
        if xi.ndim <= 2:  # simple matrix case
            # calculated with einsum_path
            return np.matmul(self.rv_inv.T*self.rv, xi)
        elif xi.shape[0] == self.rv.shape[0]:
            return np.einsum('ij, j..., ji -> i...', self.rv, xi, self.rv_inv)
        else:
            raise ValueError("Shape of xi does not match:", xi.shape)

    def _reconstruct_full(self, xi):
        if xi.ndim == 1:  # simple matrix case
            return np.matmul(self.rv*xi, self.rv_inv)
        elif xi.shape[0] == self.rv.shape[0]:
            return np.einsum('ij, j..., jk -> ik...', self.rv, xi, self.rv_inv)
        else:
            raise ValueError("Shape of xi does not match:", xi.shape)

    def _reconstruct_einsum(self, xi, sum_str):
        return np.einsum(sum_str, self.rv, xi, self.rv_inv)

    def apply(self, func, *args, **kwds):
        """Modify `self.xi` according to `func`.

        Simple wrapper to transform the eigenvalues `self.xi`.
        """
        self.xi = func(self.xi, *args, **kwds)

    def __iter__(self):
        """Allow unpacking of the attributes."""
        for attr in self.__slots__:
            yield getattr(self, attr)

    def __getitem__(self, key):
        """Make `Decomposition` behave like the tuple `rv, xi, rv_inv`."""
        return (self.rv, self.xi, self.rv_inv)[key]


def decompose_gf_omega(g_inv):
    r"""Decompose the inverse Green's function into eigenvalues and eigenvectors.

    The similarity transformation:

    .. math::
        G^{-1} = P h P^{-1}, \quad h = diag(λ(G))

    Parameters
    ----------
    g_inv : (N, N) complex ndarray
        matrix to be decomposed

    Returns
    -------
    Decomposition.rv_inv : (N, N) complex ndarray
        The *inverse* of the right eigenvectors :math:`P`
    Decomposition.h : (N) complex ndarray
        The complex eigenvalues of `g_inv`
    Decomposition.rv : (N, N) complex ndarray
        The right eigenvectors :math:`P`

    """
    h, rv = la.eig(g_inv)
    return Decomposition(rv=rv, xi=h, rv_inv=la.inv(rv))


def decompose_hamiltonian(hamilton):
    r"""Decompose the Hamiltonian matrix into eigenvalues and eigenvectors.

    The similarity transformation:

    .. math::
        H = U^\dagger h U^\dagger, \quad h = diag(λ(G))

    Parameters
    ----------
    hamilton : (N, N) complex ndarray
        matrix to be decomposed

    Returns
    -------
    Decomposition.rv_inv : (N, N) complex ndarray
        The *inverse* of the right eigenvectors :math:`U^†`. The Hamiltonian is
        hermitian, thus the decomposition is unitary :math:`U^† = U ^{-1}`
    Decomposition.h : (N) float ndarray
        The eigenvalues of `hamilton`
    Decomposition.rv : (N, N) complex ndarray
        The right eigenvectors :math:`U`

    """
    h, rv = la.eigh(hamilton)
    return Decomposition(rv=rv, xi=h, rv_inv=rv.conj().T)


def construct_gf_omega(rv, diag_inv, rv_inv):
    r"""Construct Green's function from decomposition of its inverse.

    .. math::
        G^{−1} = P h P^{-1} ⇒ G = P h^{-1} P^{-1}

    Parameters
    ----------
    rv_inv : (N, N) complex ndarray
        The inverse of the matrix of right eigenvectors (:math:`P^{-1}`)
    diag_inv : (N) array_like
        The eigenvalues (:math:`h`)
    rv : (N, N) complex ndarray
        The matrix of right eigenvectors (:math:`P`)

    Returns
    -------
    gf_omega : (N, N) complex ndarray
        The Green's function

    """
    return rv.dot(np.diagflat(diag_inv)).dot(rv_inv)
