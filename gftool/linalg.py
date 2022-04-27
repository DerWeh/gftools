"""Collection of linear algebra algorithms not contained in numpy or scipy."""
from functools import partial

import numpy as np

from scipy import linalg as spla

from gftool._util import _gu_sum

transpose = partial(np.swapaxes, axis1=-2, axis2=-1)


def orth_compl(mat):
    """Calculate the orthogonal complement of a rectangular matrix using QR.

    For a tall N×M, N>M, matrix `mat` the complete QR-decomposition gives

    .. math:: A = QR = (Q_1, Q_2) {(R_1, 0)}^2.

    The N×(N-M) matrix `Q_2` gives the orthogonal complement `A_⟂ = Q_2.T.conj()`
    with the property

    .. math:: A_⟂ A = 0.

    Parameters
    ----------
    mat : (N, M) complex array_like
        Tall input matrix.

    Returns
    -------
    mat_perp : (N-M, N) complex np.ndarray
        orthogonal complement of `mat`, such that `mat_perp@mat=0`.

    Examples
    --------
    >>> RNG = np.random.default_rng()
    >>> mat = RNG.random((10, 5))
    >>> mat_perp = orth_compl(mat)
    >>> np.allclose(mat_perp@mat, 0)
    True

    """
    dim0, dim1 = mat.shape[-2:]
    if dim0 <= dim1:
        raise ValueError(f"`mat` needs to be tall, (given: shape={mat.shape})")
    q_mat, _ = np.linalg.qr(mat, mode="complete")
    q2_mat = q_mat[..., dim1:]
    return transpose(q2_mat).conj()


def lstsq_ec(a, b, c, d, rcond=None):
    """Least-squares solution with equality constraint for linear matrix eq.

    Solves the equation `ax = b` with the constraint `cx = d`, where the vector
    `x` minimizes the squared Euclidean 2-norm :math:`||ax - b||^2_2`.
    Internally `np.linalg.lstsq` is used to solve the least-squares problem.
    The algorithm is taken from [golub2013]_.

    Parameters
    ----------
    a : (M, N) np.ndarray
        "Coefficient" matrix.
    b : (M) np.ndarray
        Ordinate or "dependent variable" values.
    c : (L, N) np.ndarray
        "Coefficient" matrix of the constrains with `L < M`.
    d : (L) np.ndarray
        Ordinate of the constrains with `L < M`.
    rcond : float, optional
        Cut-off ratio for small singular values of `a`.
        For the purposes of rank determination, singular values are treated
        as zero if they are smaller than `rcond` times the largest singular
        value of `a`. (default: machine precision times `max(M, N)`)

    Returns
    -------
    x : (N) np.ndarray
        Least-squares solution

    References
    ----------
    .. [golub2013] Golub, Gene H., und Charles F. Van Loan. Matrix Computations.
       JHU Press, 2013.

    """
    if a.shape[-1] == 0:
        return np.zeros(a.shape[:-2] + (0,), dtype=a.dtype)
    if c.shape[-2] == 0:  # no conditions given, do standard lstsq
        if c.shape[-2] != d.shape[-1]:
            raise ValueError("Mismatch of shapes of 'c' and 'd'. "
                             f"Expected (L, N), (L), got {c.shape}, {d.shape}.")
        return np.linalg.lstsq(a, b, rcond=None)[0]
    constrains = d.shape[-1]
    q_ct, r_ct = np.linalg.qr(transpose(c).conj(), mode='complete')
    r_ct = r_ct[..., :constrains, :]
    try:
        y = spla.solve_triangular(r_ct, d, trans='C')
    except np.linalg.LinAlgError as lin_err:  # try fall back to pseudo-inverse for singular values
        y, err, *__ = np.linalg.lstsq(transpose(r_ct).conj(), d, rcond=None)
        if not np.allclose(err, 0):
            raise np.linalg.LinAlgError from lin_err
    aq = a @ q_ct
    z = np.linalg.lstsq(aq[:, constrains:], b - _gu_sum(aq[:, :constrains]*y),
                        rcond=rcond)[0]
    return _gu_sum(q_ct*np.concatenate((y, z), axis=-1))
