"""Hermite-Padé approximants from Taylor expansion.

References
----------
.. [fasondini2019] Fasondini, M., Hale, N., Spoerer, R. & Weideman, J. A. C.
   Quadratic Padé Approximation: Numerical Aspects and Applications.
   Computer research and modeling 11, 1017–1031 (2019).
   https://doi.org/10.20537/2076-7633-2019-11-6-1017-1031

"""
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from scipy.linalg import toeplitz, matmul_toeplitz, solve_toeplitz

from gftool.basis import RatPol

Polynom = np.polynomial.polynomial.Polynomial


def pade(an, num_deg: int, den_deg: int, fast=False) -> RatPol:
    """Return the [`num_deg`/`den_deg`] Padé approximant to the polynomial `an`.

    Parameters
    ----------
    an : (L,) array_like
        Taylor series coefficients representing polynomial of order ``L-1``.
    num_deg, den_deg : int
        The order of the return approximating numerator/denominator polynomial.
        Must the sum must be at most ``L``: ``L >= n + m + 1``.
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
    Let's approximate the cubic root ``f(z) = (1 + z)**(1/3)`` by the ``[8/8]``
    Padé approximant:

    >>> from scipy.special import binom
    >>> an = binom(1/3, np.arange(8+8+1))  # Taylor of (1+x)**(1/3)
    >>> def f(z):
    ...     return np.emath.power(1+z, 1/3)

    >>> x = np.linspace(-1, 3, num=500)
    >>> pade = gt.hermpade.pade(an, num_deg=8, den_deg=8)

    >>> import matplotlib.pyplot as plt
    >>> __ = plt.plot(x, f(x), label='exact', color='black')
    >>> __ = plt.plot(x, np.polynomial.Polynomial(an)(x), '--', label='Taylor')
    >>> __ = plt.plot(x, pade.eval(x), ':', label='Pade')
    >>> __ = plt.ylim(ymin=0, ymax=2)
    >>> __ = plt.legend(loc='upper left')
    >>> plt.show()

    The Padé approximation is able to approximate the function even for larger
    ``x``.

    Using ``fast=True``, the Padé approximant more perform at using the
    Toeplitz structure. This might, however, be less accurate.

    >>> padef = gt.hermpade.pade(an, num_deg=8, den_deg=8, fast=True)
    >>> __ = plt.plot(x, abs(np.polynomial.Polynomial(an)(x) - f(x)), label='Taylor')
    >>> __ = plt.plot(x, abs(pade.eval(x) - f(x)), label='SVD')
    >>> __ = plt.plot(x, abs(padef.eval(x) - f(x)), label='Levinson')
    >>> __ = plt.legend()
    >>> plt.yscale('log')
    >>> plt.show()

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


def pader(an, num_deg: int, den_deg: int, rcond: float = 1e-14) -> RatPol:
    """Robust version of Padé approximant to polynomial `an`.

    Implements more or less [gonnet2013]_.

    Parameters
    ----------
    an : (L,) array_like
        Taylor series coefficients representing polynomial of order `L-1`
    num_deg, den_deg : int
        The order of the return approximating numerator/denominator polynomial.
        Must the sum must be at most `L`: `L >= n + m + 1`.
        Depending on `rcond` the degrees can be reduced.
    rcond : float, optional
        Cut-off ratio for small singular values. For the purposes of rank
        determination, singular values are treated as zero if they are smaller
        than `rcond` times the largest singular value. (default: 1e-14)
        The default is appropriate for round error due to machine precision.

    See also
    --------
    pade

    Returns
    -------
    p, q : Polynomial class
        The Padé approximation of the polynomial defined by `an` is
        ``p(x)/q(x)``.

    References
    ----------
    .. [gonnet2013] 1.Gonnet, P., Güttel, S. & Trefethen, L. N.
       Robust Padé Approximation via SVD. SIAM Rev. 55, 101–117 (2013).
       https://doi.org/10.1137/110853236

    Examples
    --------

    The robust version can avoid over fitting for high-order Padé approximants.
    Choosing an appropriate `rcond`, is however a delicate task in practice.
    We consider an example with random noise on the Taylor coefficients `an`:

    >>> from scipy.special import binom
    >>> deg = 50
    >>> an = binom(1/3, np.arange(2*deg + 1))  # Taylor of (1+x)**(1/3)
    >>> an += np.random.default_rng().normal(scale=1e-9, size=2*deg + 1)
    >>> def f(z):
    >>>     return np.emath.power(1+z, 1/3)

    >>> x = np.linspace(-1, 3, num=1000)
    >>> pade = gt.hermpade.pade(an, num_deg=deg, den_deg=deg)
    >>> pader = gt.hermpade.pader(an, num_deg=deg, den_deg=deg, rcond=1e-8)

    >>> import matplotlib.pyplot as plt
    >>> __ = plt.plot(x, abs(pade.eval(x) - f(x)), label='standard Padé')
    >>> __ = plt.plot(x, abs(pader.eval(x) - f(x)), label='robust Padé')
    >>> plt.yscale('log')
    >>> __ = plt.legend()
    >>> plt.show()

    """
    an = np.asarray(an)
    assert an.ndim == 1
    l_max = num_deg + den_deg + 1
    if an.size < l_max:
        raise ValueError("Order of q+p (den_deg+num_deg) must be smaller than len(an).")
    an = an[:l_max]
    # TODO: do rescaling, haven't found it in the reference
    tol = rcond * np.linalg.norm(an)
    if np.all(an[:num_deg] <= tol):  # up to tolerance function is 0
        return RatPol(Polynom([0]), Polynom([1]))
    row = np.r_[an[0], np.zeros(den_deg)]
    col = an
    while True:
        if den_deg == 0:
            pcoeff, qcoeff = an[:den_deg], np.array([1])
            break
        # top = np.r_[an[num_deg+1:num_deg+1-den_deg:-1], [0]*(den_deg-num_deg)]
        # amat = toeplitz(an[num_deg+1:], top)
        amat = toeplitz(col[:num_deg+den_deg+1], row[:den_deg+1])
        amat = amat[num_deg+1:num_deg+den_deg+1]
        assert amat.shape[-1] == amat.shape[-2] + 1, amat.shape
        s = np.linalg.svd(amat, compute_uv=False)
        rho = np.count_nonzero(s > rcond*s[0])
        # step 5
        if rho < den_deg:  # reduce degrees
            num_deg, den_deg = num_deg - (den_deg - rho), rho
            an = an[:num_deg + den_deg + 1]
            # tol = rcond * np.linalg.norm(an)
            # print(num_deg, den_deg, rcond*s[0])
            continue
        # TODO: code uses weird QR calculation which I don't understand
        _, s, vh = np.linalg.svd(amat)
        qcoeff = vh[-1].conj()
        assert qcoeff.size == den_deg + 1
        pcoeff = matmul_toeplitz((an[:num_deg+1], np.zeros(den_deg+1)), qcoeff)
        break
    leading_zeros = np.argmax(qcoeff != 0)
    pcoeff, qcoeff = pcoeff[leading_zeros:], qcoeff[leading_zeros:]
    trailing_zerosq = np.argmax(qcoeff[::1] != 0)
    if trailing_zerosq:
        qcoeff = qcoeff[:-trailing_zerosq]
    trailing_zerosp = np.argmax(pcoeff[::1] != 0) + 1
    if trailing_zerosq:
        pcoeff = pcoeff[:-trailing_zerosp]
    # we skip normalization of `b[0] = 1`
    return RatPol(Polynom(pcoeff), Polynom(qcoeff))


def hermite_sqr_eq(an, r_deg: int, q_deg: int, p_deg: int
                   ) -> Tuple[Polynom, Polynom, Polynom]:
    """Return the polynomials `r`, `q`, `p` for the quadratic Hermite-Padé."""
    an = np.asarray(an)
    assert an.ndim == 1
    l_max = r_deg + q_deg + p_deg + 1
    if an.size < l_max:
        raise ValueError("Order of r+q+p (r_deg+q_deg+p_deg) must be smaller than len(an).")
    an = an[:l_max]
    full_amat = toeplitz(an, r=np.zeros_like(an))
    amat2 = (full_amat@full_amat)[:, :r_deg+1]
    amat = full_amat[:, :q_deg+1]
    lower = np.concatenate((amat[p_deg+1:, :], amat2[p_deg+1:, :]), axis=-1)
    _, _, vh = np.linalg.svd(lower)
    qr = vh[-1].conj()
    assert qr.size == r_deg + q_deg + 2
    upper = np.concatenate((amat[:p_deg+1, :], amat2[:p_deg+1, :]), axis=-1)
    pcoeff = -upper@qr
    return Polynom(qr[q_deg+1:]), Polynom(qr[:q_deg+1]), Polynom(pcoeff)


@dataclass
class SqHermPadeGf(Sequence):
    """Retarded Green's function given by square Hermite-Padé approximant."""

    r: Polynom
    q: Polynom
    p: Polynom

    def eval(self, z):
        """Evaluate the retarded branch of the square Hermite-Padé approximant.

        The branch is chosen based on the imaginary part.
        """
        rz, qz, pz = self.r(z), self.q(z), self.p(z)
        discriminant = np.emath.sqrt(qz**2 - 4*pz*rz)
        # use the branch with positive spectral weight
        p_branch = 0.5*(-qz + discriminant) / rz
        m_branch = 0.5*(-qz - discriminant) / rz
        p_is_ret = p_branch.imag <= 0
        m_is_ret = m_branch.imag <= 0
        branch = np.select(
            [p_is_ret & ~m_is_ret,  # only p retarded
             ~p_is_ret & m_is_ret,  # only m retarded
             p_is_ret & m_is_ret & (p_branch.imag >= m_branch.imag),  # both retarded
             p_is_ret & m_is_ret & (m_branch.imag >= p_branch.imag),  # both retarded
             ~p_is_ret & ~m_is_ret & (p_branch.imag <= m_branch.imag),  # neither is retarded
             ~p_is_ret & ~m_is_ret & (m_branch.imag <= p_branch.imag),  # neither is retard
             ],
            [p_branch, m_branch, p_branch, m_branch, p_branch, m_branch]
        )
        return branch

    @classmethod
    def from_taylor(cls, an, deg_r: int, deg_q: int, deg_p: int):
        """Construct square Hermite-Padé from Taylor expansion `an`."""
        r, q, p = hermite_sqr_eq(an=an, r_deg=deg_r, q_deg=deg_q, p_deg=deg_p)
        return cls(r=r, q=q, p=p)

    def __getitem__(self, key: int):
        """Make `Decomposition` behave like the tuple `(rv, eig, rv_inv)`."""
        return (self.r, self.q, self.p)[key]

    def __len__(self) -> int:
        return 3
