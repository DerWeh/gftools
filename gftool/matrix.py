r"""Functions to work with Green's functions in matrix from.

A main use case of this library is the calculation of the Green's function
as the resolvent of the Hermitian or from Dyson equation.
Instead of calculating the inverse for every frequency/k-point it is oftentimes
more efficient, to calculate an eigendecomposition once.

For example, let us calculate the Green's function for 1D tight-binding chain
using:

.. math:: G(z) = [1z - H]^{-1} = [1z - UλU^†]^{-1} = U [z-λ]^{-1} U^†

>>> N = 51  # system size
>>> t = 1   # hopping amplitude
>>> hamilton = np.zeros((N, N))
>>> row, col = np.diag_indices(N)
>>> hamilton[row[:-1], col[1:]] = hamilton[row[1:], col[:-1]] = -t

>>> ww = np.linspace(-2.5, 2.5, num=201) + 1e-1j
>>> dec = gt.matrix.decompose_her(hamilton)
>>> gf_ww = dec.reconstruct(eig=1.0/(ww[:, np.newaxis] - dec.eig))

Let's check that it agrees with the inversion:

>>> gf_inv = np.linalg.inv(np.eye(N)*ww[0] - hamilton)
>>> np.allclose(gf_ww[0], gf_inv)
True

If we only need the diagonal (local) elements, we can calculate them using:

>>> gf_ww = dec.reconstruct(eig=1.0/(ww[:, np.newaxis] - dec.eig), kind='diag')

Recommended functions
---------------------

* `decompose_mat` to create `Decomposition` of general matrices
* `decompose_sym` to create `Decomposition` of complex symmetric matrices
* `decompose_her` to create `UDecomposition` of Hermitian matrices

Rest are mostly legacy functions.

"""
from __future__ import annotations

import warnings

from functools import partial
from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

transpose = partial(np.swapaxes, axis1=-2, axis2=-1)


@dataclass
class Decomposition(Sequence):
    """Decomposition of a matrix into eigenvalues and eigenvectors.

    .. math:: M = P Λ P^{-1}, Λ = diag(λ₀, λ₁, …)

    This class holds the eigenvalues and eigenvectors of the decomposition of a
    matrix and offers methods to reconstruct it.
    One intended use case is to use the `Decomposition` for the inversion of
    the Green's function to calculate it from the resolvent.

    The order of the attributes is always `rv, eig, rv_inv`, as this gives the
    reconstruct of the matrix:  `mat = (rv * eig) @ rv_inv`

    Parameters
    ----------
    rv : (..., N, N) complex np.ndarray
        The matrix of right eigenvectors.
    eig : (..., N) complex np.ndarray
        The vector of eigenvalues.
    rv_inv : (..., N, N) complex np.ndarray
        The inverse of `rv`.

    Examples
    --------
    Perform the eigendecomposition:

    >>> matrix = np.random.random((10, 10))
    >>> dec = gt.matrix.decompose_mat(matrix)
    >>> np.allclose(matrix, dec.reconstruct())
    True

    Inversion of matrix

    >>> matrix_inv = dec.reconstruct(eig=1.0/dec.eig)
    >>> np.allclose(np.linalg.inv(matrix), matrix_inv)
    True

    """

    __slots__ = ('rv', 'eig', 'rv_inv')

    rv: np.ndarray  #: The matrix of right eigenvectors.
    eig: np.ndarray  #: The vector of eigenvalues.
    rv_inv: np.ndarray  #: The inverse of `rv`.

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
    def from_gf(cls, gf) -> Decomposition:
        r"""Decompose the inverse Green's function matrix.

        The similarity transformation:

        .. math:: G^{-1} = P g P^{-1}, \quad g = diag(λ_l)

        Parameters
        ----------
        gf : (..., N, N) complex np.ndarray
            matrix to be decomposed

        Returns
        -------
        Decomposition

        """
        if isinstance(gf, cls):
            return gf
        return decompose_gf(gf)

    def reconstruct(self, eig=None, kind='full'):
        """Get matrix back from `Decomposition`.

        If the reciprocal of `self.eig` was taken, this corresponds to the
        inverse of the original matrix.

        Parameters
        ----------
        eig : (..., N) np.ndarray, optional
            Alternative value used for `self.eig`. This argument can be used
            instead of modifying `self.eig`.
        kind : {'diag', 'full'} or str
            Defines how to reconstruct the matrix. If `kind` is 'diag',
            only the diagonal elements are computed, if it is 'full' the
            complete matrix is returned.
            Alternatively a `str` used for subscript of `numpy.einsum` can be given.

        Returns
        -------
        reconstruct : (..., N, N) or (..., N) np.ndarray
            The reconstructed matrix. If a subscript string is given as `kind`,
            the shape of the output might differ.

        """
        eig = eig if eig is not None else self.eig
        kind = kind.lower()
        if 'diag'.startswith(kind):
            return ((transpose(self.rv_inv)*self.rv) @ eig[..., np.newaxis])[..., 0]
        if 'full'.startswith(kind):
            return self.rv * eig[..., np.newaxis, :] @ self.rv_inv
        return np.einsum(kind, self.rv, eig, self.rv_inv)

    def __getitem__(self, key: int):
        """Make `Decomposition` behave like the tuple `(rv, eig, rv_inv)`."""
        return (self.rv, self.eig, self.rv_inv)[key]

    def __len__(self) -> int:
        return 3

    def __str__(self) -> str:
        return f"Decomposition({self.rv.shape}x{self.eig.shape}x{self.rv_inv.shape})"


@dataclass  # pylint: disable=too-many-ancestors
class UDecomposition(Decomposition):
    """Unitary decomposition of a matrix into eigenvalues and eigenvectors.

    .. math:: H = U Λ U^†,    Λ = diag(λₗ)

    This class holds the eigenvalues and eigenvectors of the decomposition of a
    matrix and offers methods to reconstruct it.
    One intended use case is to use the `UDecomposition` for the inversion of
    the Green's function to calculate it from the resolvent.

    The order of the attributes is always `rv, eig, rv_inv`, as this gives the
    reconstruct of the matrix: `mat = (rv * eig) @ rv_inv`

    Parameters
    ----------
    rv : (..., N, N) complex np.ndarray
        The matrix of right eigenvectors.
    eig : (..., N) float np.ndarray
        The vector of real eigenvalues.
    rv_inv : (..., N, N) complex np.ndarray
        The inverse of `rv`.

    Attributes
    ----------
    u
    uh
    s

    Examples
    --------
    Perform the eigendecomposition:

    >>> matrix = np.random.random((10, 10)) + 1j*np.random.random((10, 10))
    >>> matrix = 0.5*(matrix + matrix.conj().T)
    >>> dec = gt.matrix.decompose_her(matrix)
    >>> np.allclose(matrix, dec.reconstruct())
    True

    Inversion of matrix

    >>> matrix_inv = dec.reconstruct(eig=1.0/dec.eig)
    >>> np.allclose(np.linalg.inv(matrix), matrix_inv)
    True

    The similarity transformation is unitary:

    >>> np.allclose(dec.u.conj().T, dec.uh)
    True
    >>> np.allclose(dec.u @ dec.u.conj().T, np.eye(*matrix.shape))
    True

    """

    __slots__ = ()

    @property
    def u(self):
        """Unitary matrix of right eigenvectors, same as `rv`."""
        return self.rv

    @property
    def uh(self):
        """Hermitian conjugate of unitary matrix `rv`, same as `rv_inv`."""
        return self.rv_inv

    @property
    def s(self):
        """Singular values in descending order, different from order of `eig`."""
        return np.sort(abs(self.eig))[::-1]


def decompose_mat(mat) -> Decomposition:
    r"""Decompose matrix `mat` into eigenvalues and (right) eigenvectors.

    Decompose the `mat` into `rv, eig, rv_inv`, with `mat = (rv * eig) @ rv_inv`.
    This is the similarity transformation:

    .. math:: M = P Λ P^{-1}, Λ = diag(λ₀, λ₁, …)

    where :math:`λₗ` are the eigenvalues and :math:`P` the matrix of right
    eigenvectors returned as `rv`.
    Internally, this is just a wrapper for `numpy.linalg.eig`.

    Parameters
    ----------
    mat : (..., N, N) complex np.ndarray
        matrix to be decomposed

    Returns
    -------
    Decomposition.rv : (..., N, N) complex np.ndarray
        The right eigenvectors :math:`P`
    Decomposition.eig : (..., N) complex np.ndarray
        The complex eigenvalues of `mat`
    Decomposition.rv_inv : (..., N, N) complex np.ndarray
        The *inverse* of the right eigenvectors :math:`P`

    Examples
    --------
    Perform the eigendecomposition:

    >>> matrix = np.random.random((10, 10))
    >>> rv, eig, rv_inv = gt.matrix.decompose_mat(matrix)
    >>> np.allclose(matrix, (rv * eig) @ rv_inv)
    True
    >>> np.allclose(rv @ rv_inv, np.eye(*matrix.shape))
    True

    This can also be simplified using the `Decomposition` class

    >>> dec = gt.matrix.decompose_mat(matrix)
    >>> np.allclose(matrix, dec.reconstruct())
    True

    """
    eig, vec = np.linalg.eig(mat)
    return Decomposition(rv=vec, eig=eig, rv_inv=np.linalg.inv(vec))


def decompose_sym(sym_mat, check=True) -> Decomposition:
    r"""Decompose symmetric matrix `sym_mat` into eigenvalues and (right) eigenvectors.

    Decompose the `sym_mat` into `rv, eig, rv_inv`, with `sym_mat = (rv * eig) @ rv_inv`.
    This is the *almost orthogonal* similarity transformation:

    .. math:: M = O Λ O^T,    Λ = diag(λ₀, λ₁, …)

    where :math:`λₗ` are the eigenvalues and :math:`U` the unitary matrix of
    right eigenvectors returned as `rv` with :math:`O^{-1} ≈ O^T`.
    Internally, this is just a wrapper for `numpy.linalg.eig`.
    As mentioned the transformation is only *almost orthogonal*, so you should
    not rely on this fact! Still, `decompose_sym` should be better conditioned
    than `decompose_mat` so it is preferable (so slightly slower).

    If you require orthogonality consider using [noble2017]_, it should also be
    faster.

    Parameters
    ----------
    sym_mat : (..., N, N) complex np.ndarray
        matrix to be decomposed
    check : bool, optional
        If `check`, raise an error if `sym_mat` is not symmetric. (default: True)

    Returns
    -------
    Decomposition.rv : (..., N, N) complex np.ndarray
        The right eigenvectors :math:`O`
    Decomposition.eig : (..., N) complex np.ndarray
        The complex eigenvalues of `sym_mat`
    Decomposition.rv_inv : (..., N, N) complex np.ndarray
        The *inverse* of the right eigenvectors :math:`O`

    Raises
    ------
    ValueError
        If `check=True` and `sym_mat` is not symmetric.

    References
    ----------
    .. [noble2017] Noble, J.H., Lubasch, M., Stevens, J., Jentschura, U.D., 2017.
       Diagonalization of complex symmetric matrices: Generalized Householder
       reflections, iterative deflation and implicit shifts.
       Computer Physics Communications 221, 304–316.
       https://doi.org/10.1016/j.cpc.2017.06.014

    Examples
    --------
    Perform the eigendecomposition:

    >>> matrix = np.random.random((10, 10)) + 1j*np.random.random((10, 10))
    >>> sym_mat = 0.5*(matrix + matrix.T)
    >>> rv, eig, rv_inv = gt.matrix.decompose_sym(sym_mat)
    >>> np.allclose(sym_mat, (rv * eig) @ rv_inv)
    True

    The result should be almost orthogonal, but *do not* rely on it!

    >>> np.allclose(np.linalg.inv(rv), rv.T)
    True

    This can also be simplified using the `Decomposition` class

    >>> dec = gt.matrix.decompose_sym(sym_mat)
    >>> np.allclose(sym_mat, dec.reconstruct())
    True

    """
    if check and not np.allclose(sym_mat - transpose(sym_mat), 0):
        raise ValueError("Matrix `sym_mat` is not symmetric.")
    h, rv = np.linalg.eig(sym_mat)
    # improve coordination number for complex symmetric matrices, rv_scaled is orthogonal
    rv *= np.sum(rv**2, axis=-2, keepdims=True)**-0.5
    # transpose(rv) yields same result, but is susceptible to slight asymmetry of `sym_mat`
    # np.linalg.inv is therefore more stable
    rv_inv = np.linalg.inv(rv)
    return Decomposition(rv=rv, eig=h, rv_inv=rv_inv)


def decompose_her(her_mat, check=True) -> UDecomposition:
    r"""Decompose Hermitian matrix `her_mat` into eigenvalues and (right) eigenvectors.

    Decompose the `her_mat` into `rv, eig, rv_inv`, with `her_mat = (rv * eig) @ rv_inv`.
    This is the unitary similarity transformation:

    .. math:: M = U Λ U^†,    Λ = diag(λ₀, λ₁, …)

    where :math:`λₗ` are the eigenvalues and :math:`U` the unitary matrix of
    right eigenvectors returned as `rv` with :math:`U^{-1} = U^†`.
    Internally, this is just a wrapper for `numpy.linalg.eigh`.

    Parameters
    ----------
    her_mat : (..., N, N) complex np.ndarray
        matrix to be decomposed
    check : bool, optional
        If `check`, raise an error if `her_mat` is not Hermitian. (default: True)

    Returns
    -------
    Decomposition.rv : (..., N, N) complex np.ndarray
        The right eigenvectors :math:`U`
    Decomposition.eig : (..., N) complex np.ndarray
        The complex eigenvalues of `her_mat`
    Decomposition.rv_inv : (..., N, N) complex np.ndarray
        The *inverse* of the right eigenvectors :math:`U`

    Raises
    ------
    ValueError
        If `check=True` and `her_mat` is not Hermitian.

    Examples
    --------
    Perform the eigendecomposition:

    >>> matrix = np.random.random((10, 10)) + 1j*np.random.random((10, 10))
    >>> her_mat = 0.5*(matrix + matrix.conj().T)
    >>> rv, eig, rv_inv = gt.matrix.decompose_her(her_mat)
    >>> np.allclose(her_mat, (rv * eig) @ rv_inv)
    True
    >>> np.allclose(rv @ rv.conj().T, np.eye(*her_mat.shape))
    True

    This can also be simplified using the `Decomposition` class

    >>> dec = gt.matrix.decompose_her(her_mat)
    >>> np.allclose(her_mat, dec.reconstruct())
    True

    """
    if check and not np.allclose(her_mat - transpose(her_mat.conj()), 0):
        raise ValueError("Matrix `her_mat` is not Hermitian.")
    eig, vec = np.linalg.eigh(her_mat)
    return UDecomposition(rv=vec, eig=eig, rv_inv=transpose(vec.conj()))


def decompose_gf(g_inv) -> Decomposition:
    r"""Decompose the inverse Green's function into eigenvalues and eigenvectors.

    .. deprecated:: 0.10.0
       Use the function `decompose_mat` or `decompose_sym` instead.

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
    warnings.warn("`decompose_gf` is deprecated; use `decompose_mat` or `decompose_sym` instead.",
                  category=DeprecationWarning)
    return decompose_mat(mat=g_inv)


def decompose_hamiltonian(hamilton) -> UDecomposition:
    r"""Decompose the Hamiltonian matrix into eigenvalues and eigenvectors.

    .. deprecated:: 0.10.0
       Use the function `decompose_her`.

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
    warnings.warn("`decompose_hamiltonian` is deprecated; use `decompose_her` instead.",
                  category=DeprecationWarning)
    return decompose_her(hamilton, check=False)


def construct_gf(rv, diag_inv, rv_inv):
    r"""Construct Green's function from decomposition of its inverse.

    .. math:: G^{−1} = P h P^{-1} ⇒ G = P h^{-1} P^{-1}

    It is recommended to directly use `Decomposition.reconstruct` instead.

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


def gf_2x2_z(z, eps0, eps1, hopping, hilbert_trafo=None):
    """Calculate the diagonal Green's function elements for a 2x2 system.

    Parameters
    ----------
    z : (...) complex array_like
        Complex frequencies.
    eps0, eps1 : (...) float or complex array_like
        On-site energy of element 0 and 1. For interacting systems this can be
        replaced by on-site energy + self-energy.
    hopping : (...) float or complex array_like
        Hopping element between element 0 and 1.
    hilbert_trafo : Callable, optional
        Hilbert transformation. If given, return the local Green's function.
        Else the lattice dispersion :math:`ϵ_k` can be given via `z → z - ϵ_k`.

    Returns
    -------
    gf_2x2_z : (..., 2) complex array_like
        Diagonal elements of the Green's function of the 2x2 system.

    Notes
    -----
    For the trivial case `eps0==eps1 and hopping==0`, this implementation fails.

    """
    mean_eps = np.mean([eps0, eps1], axis=0)
    sqrt_ = np.lib.scimath.sqrt(0.25*(eps0 - eps1)**2 + hopping*np.conj(hopping))
    if hilbert_trafo is None:
        gf_p, gf_m = 1./(z - mean_eps - sqrt_), 1./(z - mean_eps + sqrt_)
    else:
        gf_p, gf_m = hilbert_trafo(z - mean_eps - sqrt_), hilbert_trafo(z - mean_eps + sqrt_)
    gf0 = 0.5 / sqrt_ * ((eps0 - mean_eps + sqrt_)*gf_p - (eps0 - mean_eps - sqrt_)*gf_m)
    gf1 = 0.5 / sqrt_ * ((eps1 - mean_eps + sqrt_)*gf_p - (eps1 - mean_eps - sqrt_)*gf_m)
    return np.stack([gf0, gf1], axis=-1)
