"""Hermite-Padé approximants from Taylor expansion.

References
----------
.. [fasondini2019] Fasondini, M., Hale, N., Spoerer, R. & Weideman, J. A. C.
   Quadratic Padé Approximation: Numerical Aspects and Applications.
   Computer research and modeling 11, 1017–1031 (2019).
   https://doi.org/10.20537/2076-7633-2019-11-6-1017-1031

"""
import numpy as np

from scipy.linalg import toeplitz, matmul_toeplitz, solve_toeplitz

from gftool.basis import RatPol

Polynom = np.polynomial.polynomial.Polynomial


def pade(an, num_deg: int, den_deg: int, fast=False) -> RatPol:
    """Return the [`num_deg`/`den_deg`] Padé approximant to the polynomial `an`.

    Parameters
    ----------
    an : (L,) array_like
        Taylor series coefficients representing polynomial of order `L-1`
    num_deg, den_deg : int
        The order of the return approximating numerator/denominator polynomial.
        Must the sum must be at most `L`: `L >= n + m + 1`.
    fast : bool, optional
        If `fast`, use faster `~scipy.linalg.solve_toeplitz` algorithm.
        Else use SVD and calculate null-vector (default: False).

    Returns
    -------
    p, q : Polynomial class
        The Padé approximation of the polynomial defined by `an` is
        ``p(x)/q(x)``.

    Examples
    --------
    >>> e_exp = [1.0, 1.0, 1.0/2.0, 1.0/6.0, 1.0/24.0, 1.0/120.0]
    >>> p, q = gt.hermpade.pade(e_exp, den_deg=2, num_deg=len(e_exp)-3)

    >>> e_poly = np.poly1d(e_exp[::-1])

    Compare ``e_poly(x)`` and the Pade approximation ``p(x)/q(x)``

    >>> np.exp(1)
    2.718281828459045
    >>> e_poly(1)
    2.7166666666666668
    >>> p(1)/q(1)
    2.7179487179487181

    >>> import matplotlib.pyplot as plt
    >>> xx = np.linspace(0, 5, num=100)
    >>> __, plt.plot(xx, np.exp(xx))
    >>> __, plt.plot(xx, e_poly(xx), '--')
    >>> __, plt.plot(xx, p(xx)/q(xx), ':')

    """
    # TODO: allow to fix asymptotic by fixing `p[-1]`
    an = np.asarray(an)
    assert an.ndim == 1
    # we don't use `solve_toeplitz` as it is called less stable in the scipy doc
    l_max = num_deg + den_deg + 1
    if an.size < l_max:
        raise ValueError("Order of q+p (den_deg+num_deg) must be smaller than len(an).")
    an = an[:l_max]
    if den_deg == 0:  # trival case: no rational polynomial
        return RatPol(Polynom(an), Polynom(np.array([1])))
    # first solve the Toeplitz system for q, first row contains tailing zeros
    top = np.r_[an[num_deg+1::-1][:den_deg+1], [0]*(den_deg-num_deg-1)]
    if fast:  # use sparseness of Toeplitz matrix, we set q[N] = 1
        qcoeff = np.r_[solve_toeplitz((an[num_deg+1:], top[:-1]), b=-top[:0:-1]), 1]
    else:  # build full matrix and determine null-vector
        amat = toeplitz(an[num_deg+1:], top)
        __, __, vh = np.linalg.svd(amat)
        qcoeff = vh[-1].conj()
    assert qcoeff.size == den_deg + 1
    pcoeff = matmul_toeplitz((an[:num_deg+1], np.zeros(den_deg+1)), qcoeff)
    return RatPol(Polynom(pcoeff), Polynom(qcoeff))
