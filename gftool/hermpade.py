"""Hermite-Padé approximants from Taylor expansion."""
from typing import Tuple

import numpy as np

# requires 1.6.0
from scipy.linalg import toeplitz, matmul_toeplitz

Polynomial = np.polynomial.polynomial.Polynomial


def pade(an, num_deg: int, den_deg: int) -> Tuple[Polynomial, Polynomial]:
    """Return the [`num_deg`/`den_deg`] Padé approximant to the polynomial `an`.

    We set the coefficient `q[N] = 1`

    Parameters
    ----------
    an : (L,) array_like
        Taylor series coefficients representing polynomial of order `L-1`
    num_deg, den_deg : int
        The order of the return approximating numerator/denominator polynomial.
        Must the sum must be at most `L`: `L >= n + m + 1`.

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
    # first solve the Toeplitz system for q, we set q[N] = 1
    rhs = np.r_[np.zeros(den_deg), -an[:-den_deg]]
    amat = toeplitz(an[num_deg+1:], an[num_deg+1:num_deg+1-den_deg:-1])
    qcoeff = np.linalg.solve(amat, rhs[num_deg+1:])
    qcoeff = np.r_[qcoeff, 1]
    assert qcoeff.size == den_deg + 1
    pcoeff = matmul_toeplitz((an[:num_deg+1], np.zeros(den_deg+1)), qcoeff)
    return Polynomial(pcoeff), Polynomial(qcoeff)
