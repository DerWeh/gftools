r"""Hermite-Padé approximants from Taylor expansion.

See [fasondini2019]_ for practical applications and [baker1996]_ for the
extensive theoretical basis.

We present the example from [fasondini2019]_ showing the approximations.
We consider the cubic root ``f(z) = (1 + z)**(1/3)``, the radius of convergence
of its series is 1.

Taylor series
~~~~~~~~~~~~~
Obviously the Taylor series fails for `z<=1` as it cannot represent a pole,
but also for larger `z>=1` it fails:

.. plot::
   :format: doctest
   :context: close-figs

   >>> from scipy.special import binom
   >>> an = binom(1/3, np.arange(17))  # Taylor of (1+x)**(1/3)
   >>> def f(z):
   ...     return np.emath.power(1+z, 1/3)
   >>> taylor = np.polynomial.Polynomial(an)

   >>> import matplotlib.pyplot as plt
   >>> x = np.linspace(0, 3, num=500)
   >>> __ = plt.plot(x, f(x), color='black')
   >>> __ = plt.plot(x, taylor(x), color='C1')
   >>> __ = plt.ylim(0, 1.75)
   >>> plt.show()

Padé approximant
~~~~~~~~~~~~~~~~
The Padé approximant can be used to improve the Taylor expansion and expands
the applicability beyond the radius of convergence:

.. plot::
   :format: doctest
   :context: close-figs

   >>> x = np.linspace(-3, 3, num=501)
   >>> pade = gt.hermpade.pade(an, num_deg=8, den_deg=8)
   >>> __ = plt.plot(x, f(x).real, color='black')
   >>> __ = plt.plot(x, pade.eval(x), color='C1')
   >>> __ = plt.ylim(0, 1.75)
   >>> plt.show()

The Padé approximant provides a global approximation.
For negative values, however, the Padé approximant still fails, as it cannot
accurately represent a branch cut.  The Padé approximant is suitable for simple
poles and tries to approximate the branch-cut by a finite number of poles.
It is instructive to plot the error in the complex plane:

.. plot::
   :format: doctest
   :context: close-figs

   >>> y = np.linspace(-3, 3, num=501)
   >>> z = x[:, None] + 1j*y[None, :]
   >>> error = abs(pade.eval(z) - f(z))

   >>> import matplotlib as mpl
   >>> fmt = mpl.ticker.LogFormatterMathtext()
   >>> __ = fmt.create_dummy_axis()
   >>> norm = mpl.colors.LogNorm(vmin=1e-16, vmax=1)
   >>> __ = plt.pcolormesh(x, y, error.T, shading='nearest', norm=norm)
   >>> cbar = plt.colorbar(extend='both')
   >>> levels = np.logspace(-15, 0, 16)
   >>> cont = plt.contour(x, y, error.T, colors='black', linewidths=0.25, levels=levels)
   >>> __ = plt.clabel(cont, cont.levels, fmt=fmt, fontsize='x-small')
   >>> for ll in levels:
   ...     __ = cbar.ax.axhline(ll, color='black', linewidth=0.25)
   >>> __ = plt.xlabel(r"$\Re z$")
   >>> __ = plt.ylabel(r"$\Im z$")
   >>> plt.tight_layout()
   >>> plt.gca().set_rasterization_zorder(1.5)  # avoid excessive files
   >>> plt.show()

Away from the branch-cut, the Padé approximant is a reasonable approximation.


Square Hermite-Padé approximant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A further improvement is obtained by using the square Hermite-Padé approximant,
which can represent square-root branch cuts:

.. plot::
   :format: doctest
   :context: close-figs

   >>> herm2 = gt.hermpade.Hermite2.from_taylor(an, deg_p=5, deg_q=5, deg_r=5)

   >>> __ = plt.plot(x, f(x).real, color='black')
   >>> __ = plt.plot(x, herm2.eval(x).real, color='C1')
   >>> __ = plt.ylim(0, 1.75)
   >>> plt.show()

It nicely approximates the function almost everywhere. Let's compare the error
to the Padé approximant:

.. plot::
   :format: doctest
   :context: close-figs

   >>> __ = plt.plot(x, abs(pade.eval(x) - f(x)), label="Padé")
   >>> __ = plt.plot(x, abs(herm2.eval(x) - f(x)), label="Herm2")
   >>> __ = plt.yscale('log')
   >>> __ = plt.legend()
   >>> plt.show()

Let's also compare the quality of the approximants in the complex plane:

.. plot::
   :format: doctest
   :context: close-figs

   >>> error2 = np.abs(herm2.eval(z) - f(z))

   >>> __, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
   >>> __ = axes[0].set_title("Padé")
   >>> __ = axes[1].set_title("Herm2")
   >>> for ax, err in zip(axes, [error, error2]):
   ...     pcm = ax.pcolormesh(x, y, err.T, shading='nearest', norm=norm)
   ...     cont = ax.contour(x, y, err.T, colors='black', linewidths=0.25, levels=levels)
   ...     __ = ax.clabel(cont, cont.levels, fmt=fmt, fontsize='x-small')
   ...     __ = ax.set_xlabel(r"$\Re z$")
   ...     ax.set_rasterization_zorder(1.5)
   >>> __ = axes[0].set_ylabel(r"$\Im z$")
   >>> plt.tight_layout()
   >>> cbar = plt.colorbar(pcm, extend='both', ax=axes, fraction=0.08, pad=0.02)
   >>> cbar.ax.tick_params(labelsize='x-small')
   >>> for ll in levels:
   ...     __ = cbar.ax.axhline(ll, color='black', linewidth=0.25)
   >>> plt.show()

Note, however, the square Hermite-Padé approximant contains the ambiguity which
branch to choose. The heuristic can fail and should therefore be checked.

References
----------
.. [fasondini2019] Fasondini, M., Hale, N., Spoerer, R. & Weideman, J. A. C.
   Quadratic Padé Approximation: Numerical Aspects and Applications.
   Computer research and modeling 11, 1017–1031 (2019).
   https://doi.org/10.20537/2076-7633-2019-11-6-1017-1031
.. [baker1996] Baker Jr, G. A. & Graves-Morris, Pade Approximants.
   Second edition. (Cambridge University Press, 1996).

"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.linalg import matmul_toeplitz, qr, solve_toeplitz, toeplitz

from gftool.basis import RatPol

Polynom = np.polynomial.polynomial.Polynomial


def _strip_ceoffs(pcoeff, qcoeff):
    """Strip leading/tailing zeros from coefficients."""
    leading_zeros = np.argmax(qcoeff != 0)
    pcoeff, qcoeff = pcoeff[leading_zeros:], qcoeff[leading_zeros:]
    trailing_zerosq = np.argmax(qcoeff[::-1] != 0)
    if trailing_zerosq:
        qcoeff = qcoeff[:-trailing_zerosq]
    trailing_zerosp = np.argmax(pcoeff[::-1] != 0)
    if trailing_zerosp:
        pcoeff = pcoeff[:-trailing_zerosp]
    return pcoeff, qcoeff


def _nullvec(mat):
    """Determine a single null-vector of `mat` using QR decomposition.

    Parameters
    ----------
    mat : (N-1, N) complex np.ndarray
        The matrix for which we calculate the null-vector.

    Returns
    -------
    nullvec : (N) complex np.ndarray
        The approximate null-vector corresponding to `mat`.

    """
    q_, __ = qr(mat.conj().T, mode='full')
    return q_[:, -1]


def _nullvec_lst(mat, fix: int, rcond=None):
    """Determine the null-vector of `mat` in a least-squares sense.

    Typically the null-vector is found as the singular vector corresponding to
    the smallest singular vector.

    Instead, we set the component `fix` to 1, and solve the equations using
    `~numpy.linalg.lstsq`.

    Parameters
    ----------
    mat : (M, N) np.ndarray
        The matrix for which we calculate the null-vector.
    fix : int
        The index of the component we fix to 1. Negative values are allowed.
    rcond : float, optional
        Cut-off ratio for small singular values of a`mat`. For the purposes of
        rank determination, singular values are treated as zero if they are
        smaller than `rcond` times the largest singular value of `mat`.
        (default: machine precision times `max(M, N)`)

    Returns
    -------
    nullvec : (N) np.ndarray
        The approximate null-vector corresponding to `mat`.

    """
    if fix < 0:  # handle negative indices as we use fix+1
        fix = mat.shape[-1] + fix
    if fix >= mat.shape[-1]:
        raise ValueError
    vec, *__ = np.linalg.lstsq(
        np.concatenate((mat[:, :fix], mat[:, fix+1:]), axis=-1),
        -mat[:, fix], rcond=rcond,
    )
    vec = np.r_[vec[:fix], 1, vec[fix:]]
    return vec


def pade(an, num_deg: int, den_deg: int, fast=False) -> RatPol:
    """Return the [`num_deg`/`den_deg`] Padé approximant to the polynomial `an`.

    Parameters
    ----------
    an : (L,) array_like
        Taylor series coefficients representing polynomial of order ``L-1``.
    num_deg, den_deg : int
        The order of the return approximating numerator/denominator polynomial.
        The sum must be at most ``L``: ``L >= num_deg + den_deg + 1``.
    fast : bool, optional
        If `fast`, use faster `~scipy.linalg.solve_toeplitz` algorithm.
        Else use QR and calculate null-vector (default: False).

    Returns
    -------
    RatPol
        The rational polynomial with numerator `RatPol.numer`,
        and denominator `RatPol.denom`.

    Examples
    --------
    Let's approximate the cubic root ``f(z) = (1 + z)**(1/3)`` by the ``[8/8]``
    Padé approximant:

    >>> from scipy.special import binom
    >>> an = binom(1/3, np.arange(8+8+1))  # Taylor of (1+x)**(1/3)
    >>> x = np.linspace(-1, 3, num=500)
    >>> fx = np.emath.power(1+x, 1/3)

    >>> pade = gt.hermpade.pade(an, num_deg=8, den_deg=8)

    >>> import matplotlib.pyplot as plt
    >>> __ = plt.plot(x, fx, label='exact', color='black')
    >>> __ = plt.plot(x, np.polynomial.Polynomial(an)(x), '--', label='Taylor')
    >>> __ = plt.plot(x, pade.eval(x), ':', label='Pade')
    >>> __ = plt.ylim(ymin=0, ymax=2)
    >>> __ = plt.legend(loc='upper left')
    >>> plt.show()

    The Padé approximation is able to approximate the function even for larger
    ``x``.

    Using ``fast=True``, the Toeplitz structure is used to evaluate the `pade`
    faster using Levinson recursion.  This might, however, be less accurate in
    some cases.

    >>> padef = gt.hermpade.pade(an, num_deg=8, den_deg=8, fast=True)
    >>> __ = plt.plot(x, abs(np.polynomial.Polynomial(an)(x) - fx), label='Taylor')
    >>> __ = plt.plot(x, abs(pade.eval(x) - fx), label='QR')
    >>> __ = plt.plot(x, abs(padef.eval(x) - fx), label='Levinson')
    >>> __ = plt.legend()
    >>> plt.yscale('log')
    >>> plt.show()

    """
    # TODO: allow to fix asymptotic by fixing `p[-1]`
    an = np.asarray(an)
    assert an.ndim == 1
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
        qcoeff = _nullvec(amat)
    assert qcoeff.size == den_deg + 1
    pcoeff = matmul_toeplitz((an[:num_deg+1], np.zeros(den_deg+1)), qcoeff)
    return RatPol(numer=Polynom(pcoeff), denom=Polynom(qcoeff))


def pade_lstsq(an, num_deg: int, den_deg: int, rcond=None, fix_q=None) -> RatPol:
    """Return the [`num_deg`/`den_deg`] Padé approximant to the polynomial `an`.

    Same as `pade`, however all elements of `an` are taken into account.
    Instead of finding the null-vector of the underdetermined system,
    the parameter ``RatPol.denom.coeff[0]=1`` is fixed and the system is solved
    truncating small singular values.

    Parameters
    ----------
    an : (L,) array_like
        Taylor series coefficients representing polynomial of order ``L-1``.
    num_deg, den_deg : int
        The order of the return approximating numerator/denominator polynomial.
        The sum must be at most ``L``: ``L >= num_deg + den_deg + 1``.
    rcond : float, optional
        Cut-off ratio for small singular values for the denominator polynomial.
        For the purposes of rank determination, singular values are treated
        as zero if they are smaller than `rcond` times the largest singular
        value. (default: machine precision times `den_deg`)

    Returns
    -------
    RatPol
        The rational polynomial with numerator `RatPol.numer`,
        and denominator `RatPol.denom`.

    See Also
    --------
    pade
    numpy.linalg.lstsq

    """
    an = np.asarray(an)
    assert an.ndim == 1
    l_max = num_deg + den_deg + 1
    if an.size < l_max:
        raise ValueError("Order of q+p (den_deg+num_deg) must be smaller than len(an).")
    if den_deg == 0:  # trivial case: no rational polynomial
        return RatPol(Polynom(an), Polynom(np.array([1])))
    # first solve the Toeplitz system for q, first row contains tailing zeros
    top = np.r_[an[num_deg+1::-1][:den_deg+1], [0]*(den_deg-num_deg-1)]
    amat = toeplitz(an[num_deg+1:], top)
    if fix_q is None:
        _, _, vh = np.linalg.svd(amat)
        qcoeff = vh[0].conj()
        fix_q = np.argmin(abs(qcoeff))
    qcoeff = _nullvec_lst(amat, fix=0, rcond=rcond)
    assert qcoeff.size == den_deg + 1
    pcoeff = matmul_toeplitz((an[:num_deg+1], np.zeros(den_deg+1)), qcoeff)
    return RatPol(numer=Polynom(pcoeff), denom=Polynom(qcoeff))


def pader(an, num_deg: int, den_deg: int, rcond: float = 1e-14) -> RatPol:
    """Robust version of Padé approximant to polynomial `an`.

    Implements more or less [gonnet2013]_. The degrees `num_deg` and `den_deg`
    are automatically reduced to obtain a robust solution.

    Parameters
    ----------
    an : (L,) array_like
        Taylor series coefficients representing polynomial of order ``L-1``
    num_deg, den_deg : int
        The order of the return approximating numerator/denominator polynomial.
        The sum must be at most ``L``: ``L >= n + m + 1``.
        Depending on `rcond` the degrees can be reduced.
    rcond : float, optional
        Cut-off ratio for small singular values. For the purposes of rank
        determination, singular values are treated as zero if they are smaller
        than `rcond` times the largest singular value. (default: 1e-14)
        The default is appropriate for round error due to machine precision.

    Returns
    -------
    RatPol
        The rational polynomial with numerator `RatPol.numer`,
        and denominator `RatPol.denom`.

    See also
    --------
    pade

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
    >>> x = np.linspace(-1, 3, num=500)
    >>> fx = np.emath.power(1+x, 1/3)

    >>> pade = gt.hermpade.pade(an, num_deg=deg, den_deg=deg)
    >>> pader = gt.hermpade.pader(an, num_deg=deg, den_deg=deg, rcond=1e-8)

    >>> import matplotlib.pyplot as plt
    >>> __ = plt.plot(x, abs(pade.eval(x) - fx), label='standard Padé')
    >>> __ = plt.plot(x, abs(pader.eval(x) - fx), label='robust Padé')
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
    if np.all(abs(an[:num_deg]) <= tol):  # up to tolerance function is 0
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
        qcoeff = _nullvec(amat)
        assert qcoeff.size == den_deg + 1
        pcoeff = matmul_toeplitz((an[:num_deg+1], np.zeros(den_deg+1)), qcoeff)
        break
    pcoeff, qcoeff = _strip_ceoffs(pcoeff=pcoeff, qcoeff=qcoeff)
    # we skip normalization of `b[0] = 1`
    return RatPol(Polynom(pcoeff), Polynom(qcoeff))


def hermite2(an, p_deg: int, q_deg: int, r_deg: int) -> Tuple[Polynom, Polynom, Polynom]:
    r"""Return the polynomials `p`, `q`, `r` for the quadratic Hermite-Padé approximant.

    The polynomials fulfill the equation

    .. math:: p(x) + q(x) f(x) + r(x) f^2(x) = 𝒪(x^{N_p + N_q + N_r + 2})

    where :math:`f(x)` is the function with Taylor coefficients `an`,
    and :math:`N_x` are the degrees of the polynomials.
    The approximant has two branches

    .. math:: F^±(z) = [-q(z) ± \sqrt{q^2(z) - 4p(z)r(z)}] / 2r(z)

    Parameters
    ----------
    an : (L,) array_like
        Taylor series coefficients representing polynomial of order ``L-1``.
    p_deg, q_deg, r_deg : int
        The order of the polynomials of the quadratic Hermite-Padé approximant.
        The sum must be at most ``p_deg + q_deg + r_deg + 2 <= L``.

    Returns
    -------
    p, q, r : Polynom
        The polynomials `p`, `q`, and `r` building the quadratic Hermite-Padé
        approximant.

    See Also
    --------
    Hermite2 : high level interface, guessing the correct branch

    Examples
    --------
    The quadratic Hermite-Padé approximant can reproduce the square root
    ``f(z) = (1 + z)**(1/2)``:

    >>> from scipy.special import binom
    >>> an = binom(1/2, np.arange(5+5+5+2))  # Taylor of (1+x)**(1/2)
    >>> x = np.linspace(-3, 3, num=500)
    >>> fx = np.emath.power(1+x, 1/2)

    >>> p, q, r = gt.hermpade.hermite2(an, 5, 5, 5)
    >>> px, qx, rx = p(x), q(x), r(x)
    >>> pos_branch = (-qx + np.emath.sqrt(qx**2 - 4*px*rx)) / (2*rx)

    >>> import matplotlib.pyplot as plt
    >>> __ = plt.plot(x, fx.real, label='exact', color='black')
    >>> __ = plt.plot(x, fx.imag, '--', color='black')
    >>> __ = plt.plot(x, pos_branch.real, '--', label='Herm2', color='C1')
    >>> __ = plt.plot(x, pos_branch.imag, ':', color='C1')
    >>> plt.show()

    """
    an = np.asarray(an)
    assert an.ndim == 1
    l_max = r_deg + q_deg + p_deg + 2
    if an.size < l_max:
        raise ValueError("Order of r+q+p (r_deg+q_deg+p_deg) must be smaller than len(an).")
    an = an[:l_max]
    full_amat = toeplitz(an, r=np.zeros_like(an))
    amat2 = (full_amat@full_amat[:, :r_deg+1])
    amat = full_amat[:, :q_deg+1]
    lower = np.concatenate((amat[p_deg+1:, :], amat2[p_deg+1:, :]), axis=-1)
    qrcoeff = _nullvec(lower)
    assert qrcoeff.size == r_deg + q_deg + 2
    upper = np.concatenate((amat[:p_deg+1, :], amat2[:p_deg+1, :]), axis=-1)
    pcoeff = -upper@qrcoeff
    return Polynom(pcoeff), Polynom(qrcoeff[:q_deg+1]), Polynom(qrcoeff[q_deg+1:])


def hermite2_lstsq(an, p_deg: int, q_deg: int, r_deg: int,
                   rcond=None, fix_qr=None) -> Tuple[Polynom, Polynom, Polynom]:
    r"""Return the polynomials `p`, `q`, `r` for the quadratic Hermite-Padé approximant.

    Same as `hermite2`, however all elements of `an` are taken into account.
    Instead of finding the null-vector of the underdetermined system,
    the parameter ``q.coeff[0]=1`` is fixed and the system is solved truncating
    small singular values.

    The polynomials fulfill the equation

    .. math:: p(x) + q(x) f(x) + r(x) f^2(x) = 𝒪(x^{N_p + N_q + N_r + 2})

    where :math:`f(x)` is the function with Taylor coefficients `an`,
    and :math:`N_x` are the degrees of the polynomials.
    The approximant has two branches

    .. math:: F^±(z) = [-q(z) ± \sqrt{q^2(z) - 4p(z)r(z)}] / 2r(z)

    Parameters
    ----------
    an : (L,) array_like
        Taylor series coefficients representing polynomial of order ``L-1``.
    p_deg, q_deg, r_deg : int
        The order of the polynomials of the quadratic Hermite-Padé approximant.
        The sum must be at most ``p_deg + q_deg + r_deg + 2 <= L``.
    rcond : float, optional
        Cut-off ratio for small singular values. For the purposes of rank
        determination, singular values are treated as zero if they are
        smaller than `rcond` times the largest singular value.
        (default: machine precision times maximum of matrix dimensions)
    fix_qr : int, optional
        The coefficient which is fixed to 1. The values ``0 <= fix_qr <= q_deg``
        corresponds to the coefficients of the polynomial `q`,
        the values ``q_deg + 1 <= fix_qr <= q_deg + r_deg + 1`` correspond to
        the coefficients of the polynomial `r`.

    Returns
    -------
    p, q, r : Polynom
        The polynomials `p`, `q`, and `r` building the quadratic Hermite-Padé
        approximant.

    See Also
    --------
    hermite2
    Hermite2 : high level interface, guessing the correct branch
    numpy.linalg.lstsq

    """
    an = np.asarray(an)
    assert an.ndim == 1
    if an.size < r_deg + q_deg + p_deg + 2:
        raise ValueError("Order of r+q+p (r_deg+q_deg+p_deg) must be smaller than len(an).")
    if np.all(an == 0):  # cannot handle this edge case
        return Polynom([0]*(p_deg+1)), Polynom([0]*(q_deg+1)), Polynom([1]+[0]*r_deg)
    full_amat = toeplitz(an, r=np.zeros_like(an))
    amat2 = (full_amat@full_amat[:, :r_deg+1])
    amat = full_amat[:, :q_deg+1]
    lower = np.concatenate((amat[p_deg+1:, :], amat2[p_deg+1:, :]), axis=-1)
    if fix_qr is None:
        _, _, vh = np.linalg.svd(lower)
        # heuristic: choose most important vector according to SVD, i.e. the
        # complete opposite of the null-vector, and fix its smallest element
        fix_qr = np.argmin(abs(vh[0]))
    qrcoeff = _nullvec_lst(lower, fix=fix_qr, rcond=rcond)
    assert qrcoeff.size == r_deg + q_deg + 2
    upper = np.concatenate((amat[:p_deg+1, :], amat2[:p_deg+1, :]), axis=-1)
    pcoeff = -upper@qrcoeff
    return Polynom(pcoeff), Polynom(qrcoeff[:q_deg+1]), Polynom(qrcoeff[q_deg+1:])


@dataclass
class _Hermite2Base:
    """Basic container for square Hermite-Padé approximant."""

    p: Polynom
    q: Polynom
    r: Polynom

    def eval_branches(self, z) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the two branches."""
        pz, qz, rz = self.p(z), self.q(z), self.r(z)
        discriminant = np.emath.sqrt(qz**2 - 4*pz*rz)
        p_branch = 0.5*(-qz + discriminant) / rz
        m_branch = 0.5*(-qz - discriminant) / rz
        p_stable = np.where(abs(p_branch) >= abs(m_branch), p_branch, pz / (rz*m_branch))
        m_stable = np.where(abs(m_branch) >= abs(p_branch), m_branch, pz / (rz*p_branch))
        return p_stable, m_stable


@dataclass
class Hermite2(_Hermite2Base):
    r"""Square Hermite-Padé approximant with branch selection according to Padé.

    A function :math:`f(z)` with known Taylor coefficients `an` is approximated
    using

    .. math:: p(z) + q(z)f(z) + r(z) f^2(z) = 𝒪(z^{N_p + N_q + N_r + 2})

    where :math:`f(z)` is the function with Taylor coefficients `an`,
    and :math:`N_x` are the degrees of the polynomials.
    The approximant has two branches

    .. math:: F^±(z) = [-q(z) ± \sqrt{q^2(z) - 4p(z)r(z)}] / 2r(z)

    The function `Hermite2.eval` chooses the branch which is locally closer
    to the Padé approximant, as proposed by [fasondini2019]_.

    Parameters
    ----------
    p, q, r : Polynom
        The polynomials.
    pade : RatPol
        The Padé approximant.

    References
    ----------
    .. [fasondini2019] Fasondini, M., Hale, N., Spoerer, R. & Weideman, J. A. C.
       Quadratic Padé Approximation: Numerical Aspects and Applications.
       Computer research and modeling 11, 1017–1031 (2019).
       https://doi.org/10.20537/2076-7633-2019-11-6-1017-1031

    Examples
    --------
    Let's approximate the cubic root ``f(z) = (1 + z)**(1/3)`` by the ``[5/5/5]``
    square Hermite-Padé approximant:

    >>> from scipy.special import binom
    >>> an = binom(1/3, np.arange(5+5+5+2))  # Taylor of (1+x)**(1/3)
    >>> x = np.linspace(-1, 2, num=500)
    >>> fx = np.emath.power(1+x, 1/3)

    >>> herm = gt.hermpade.Hermite2.from_taylor(an, 5, 5, 5)

    >>> import matplotlib.pyplot as plt
    >>> __ = plt.plot(x, fx, label='exact', color='black')
    >>> __ = plt.plot(x, np.polynomial.Polynomial(an)(x), '--', label='Taylor')
    >>> __ = plt.plot(x, herm.pade.eval(x), '-.', label='Padé')
    >>> __ = plt.plot(x, herm.eval(x).real, ':', label='Herm2')
    >>> __ = plt.ylim(ymin=0, ymax=1.75)
    >>> __ = plt.legend(loc='upper left')
    >>> plt.show()

    The improvement becomes more clear showing the error:

    >>> __ = plt.plot(x, abs(np.polynomial.Polynomial(an)(x) - fx), '--', label='Taylor')
    >>> __ = plt.plot(x, abs(herm.pade.eval(x) - fx), '-.', label='Padé')
    >>> __ = plt.plot(x, abs(herm.eval(x) - fx), ':', label='Herm2')
    >>> __ = plt.legend()
    >>> plt.yscale('log')
    >>> plt.show()

    Mind, that the predication of the correct branch is far from safe:

    >>> an = binom(1/2, np.arange(8+8+1))  # Taylor of (1+x)**(1/2)
    >>> x = np.linspace(-3, 3, num=500)
    >>> fx = np.emath.power(1+x, 1/2)
    >>> herm = gt.hermpade.Hermite2.from_taylor(an, 5, 5, 5)

    >>> __ = plt.plot(x, fx.real, label='exact', color='black')
    >>> __ = plt.plot(x, herm.eval(x).real, label='Square', color='C1')
    >>> __ = plt.plot(x, fx.imag, '--', color='black')
    >>> __ = plt.plot(x, herm.eval(x).imag, '--', color='C1')
    >>> plt.show()

    The positive branch, however, yields the exact result:

    >>> p_branch, __ = herm.eval_branches(x)
    >>> np.allclose(p_branch, fx, rtol=1e-14, atol=1e-14)
    True

    """

    p: Polynom
    q: Polynom
    r: Polynom
    pade: RatPol

    def eval(self, z):
        """Evaluate square approximant choosing branch based on Padé."""
        p_branch, m_branch = self.eval_branches(z)
        pade_ = self.pade.eval(z)
        approx = np.where(abs(p_branch - pade_) < abs(m_branch - pade_), p_branch, m_branch)
        return approx

    @classmethod
    def from_taylor(cls, an, deg_p: int, deg_q: int, deg_r: int) -> "Hermite2":
        """Construct square Hermite-Padé from Taylor expansion `an`."""
        p, q, r = hermite2(an=an, p_deg=deg_p, q_deg=deg_q, r_deg=deg_r)
        deg_diff = max(deg_q, int(np.sqrt(deg_p*deg_r))) - deg_r
        length = deg_r + deg_q + deg_p
        den_deg = (length - deg_diff) // 2
        pade_ = pade(an=an, num_deg=den_deg+deg_diff, den_deg=den_deg)
        return cls(r=r, q=q, p=p, pade=pade_)

    @classmethod
    def from_taylor_lstsq(cls, an, deg_p: int, deg_q: int, deg_r: int,
                          rcond=None, fix_qr=None) -> "Hermite2":
        """Construct square Hermite-Padé from Taylor expansion `an`."""
        p, q, r = hermite2_lstsq(an=an, p_deg=deg_p, q_deg=deg_q, r_deg=deg_r,
                                 rcond=rcond, fix_qr=fix_qr)
        deg_diff = max(deg_q, int(np.sqrt(deg_p*deg_r))) - deg_r
        length = deg_r + deg_q + deg_p
        den_deg = (length - deg_diff) // 2
        pade_ = pade_lstsq(an=an, num_deg=den_deg+deg_diff, den_deg=den_deg, rcond=rcond)
        return cls(r=r, q=q, p=p, pade=pade_)


@dataclass
class _Hermite2Ret(_Hermite2Base):
    """Retarded Green's function given by square Hermite-Padé approximant.

    .. warning:: highly experimental and will probably vanish.

    """

    def eval(self, z):
        """Evaluate the retarded branch of the square Hermite-Padé approximant.

        The branch is chosen based on the imaginary part.
        """
        p_branch, m_branch = self.eval_branches(z)
        # use the branch with positive spectral weight
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
        p, q, r = hermite2(an=an, p_deg=deg_p, q_deg=deg_q, r_deg=deg_r)
        return cls(p=p, q=q, r=r)

    @classmethod
    def from_taylor_lstsq(cls, an, deg_p: int, deg_q: int, deg_r: int,
                          rcond=None, fix_qr=None) -> "Hermite2":
        """Construct square Hermite-Padé from Taylor expansion `an`."""
        p, q, r = hermite2_lstsq(an=an, p_deg=deg_p, q_deg=deg_q, r_deg=deg_r,
                                 rcond=rcond, fix_qr=fix_qr)
        return cls(p=p, q=q, r=r)
