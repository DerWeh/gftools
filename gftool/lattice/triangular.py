r"""2D triangular lattice.

The dispersion of the 2D triangular lattice is given by

.. math:: ϵ_{k_x, k_y} = t [\cos(2k_x) + 2 \cos(k_x)\cos(k_y)]

which takes values :math:`ϵ_{k_x, k_y} ∈ [-1.5t, 3t] = [-2D/3, 4D/3]`.

:half_bandwidth: The half-bandwidth `D` corresponds to a nearest neighbor hopping
                 of `t=4D/9`.

"""
import numpy as np

from mpmath import mp
from scipy.special import ellipkm1

from gftool._util import _u_ellipk


def _signed_sqrt(z):
    """Square root with correct sign for triangular lattice."""
    sign = np.where((z.real < 0) & (z.imag < 0), -1, 1)
    return sign * np.lib.scimath.sqrt(z)


def gf_z(z, half_bandwidth):
    r"""Local Green's function of the 2D triangular lattice.

    Note, that the spectrum is asymmetric and in :math:`[-2D/3, 4D/3]`,
    where :math:`D` is the half-bandwidth.
    The Green's function is evaluated as complete elliptic integral of first
    kind, see [horiguchi1972]_.

    Parameters
    ----------
    z : complex np.ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : float
        Half-bandwidth of the DOS of the triangular lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping
        :math:`t=4D/9`.

    Returns
    -------
    gf_z : complex np.ndarray or complex
        Value of the triangular lattice Green's function

    References
    ----------
    .. [horiguchi1972] Horiguchi, T., 1972. Lattice Green’s Functions for the
       Triangular and Honeycomb Lattices. Journal of Mathematical Physics 13,
       1411–1419. https://doi.org/10.1063/1.1666155

    Examples
    --------
    >>> ww = np.linspace(-1.5, 1.5, num=500, dtype=complex) + 1e-64j
    >>> gf_ww = gt.lattice.triangular.gf_z(ww, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.axhline(0, color='black', linewidth=0.8)
    >>> _ = plt.axvline(-2/3, color='black', linewidth=0.8)
    >>> _ = plt.axvline(+4/3, color='black', linewidth=0.8)
    >>> _ = plt.plot(ww.real, gf_ww.real, label=r"$\Re G$")
    >>> _ = plt.plot(ww.real, gf_ww.imag, '--', label=r"$\Im G$")
    >>> _ = plt.ylabel(r"$G*D$")
    >>> _ = plt.xlabel(r"$\omega/D$")
    >>> _ = plt.xlim(left=ww.real.min(), right=ww.real.max())
    >>> _ = plt.legend()
    >>> plt.show()

    """
    D = half_bandwidth * 4 / 9
    z = 1.0 / D * np.asarray(z)
    shape = z.shape
    z = z.reshape(-1)
    advanced = z.imag < 0
    z = np.where(advanced, np.conj(z), z)  # calculate retarded only, and use symmetry
    singular = D * z == -1  # try singularity which needs to be avoided
    z[singular] = 0  # mock value to avoid errors
    rr = _signed_sqrt(2*z + 3)
    gg = 4.0 / (_signed_sqrt(rr - 1)**3 * _signed_sqrt(rr + 3))  # eq (2.9)
    kk = _signed_sqrt(rr) * gg  # eq (2.11)
    mm = kk**2
    K = np.asarray(_u_ellipk(mm))
    # eqs (2.22) and eq (2.18), fix correct plane
    K[kk.imag > 0] += 2j*_u_ellipk(1 - mm[kk.imag > 0])
    gf_z = 1 / np.pi / D * gg * K  # eq (2.6)
    gf_z[singular] = 0 - 1j*np.infty
    return np.where(advanced, np.conj(gf_z), gf_z).reshape(shape)  # return to advanced by symmetry


def hilbert_transform(xi, half_bandwidth):
    r"""Hilbert transform of non-interacting DOS of the triangular lattice.

    The Hilbert transform is defined

    .. math:: \tilde{D}(ξ) = ∫_{-∞}^{∞}dϵ \frac{DOS(ϵ)}{ξ − ϵ}

    The lattice Hilbert transform is the same as the non-interacting Green's
    function.

    Parameters
    ----------
    xi : complex np.ndarray or complex
        Point at which the Hilbert transform is evaluated
    half_bandwidth : float
        half-bandwidth of the DOS of the 2D triangular lattice

    Returns
    -------
    hilbert_transform : complex np.ndarray or complex
        Hilbert transform of `xi`.

    Notes
    -----
    Relation between nearest neighbor hopping `t` and half-bandwidth `D`

    .. math:: 9t = 4D

    See Also
    --------
    gftool.lattice.triangular.gf_z

    """
    return gf_z(xi, half_bandwidth)


def dos(eps, half_bandwidth):
    r"""DOS of non-interacting 2D triangular lattice.

    The DOS diverges at `-4/9*half_bandwidth`.
    The DOS is evaluated as complete elliptic integral of first kind,
    see [kogan2021]_.

    Parameters
    ----------
    eps : float np.ndarray or float
        DOS is evaluated at points `eps`.
    half_bandwidth : float
        Half-bandwidth of the DOS, DOS(`eps` < -2/3`half_bandwidth`) = 0,
        DOS(4/3`half_bandwidth` < `eps`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping
        :math:`t=4D/9`.

    Returns
    -------
    dos : float np.ndarray or float
        The value of the DOS.

    See Also
    --------
    gftool.lattice.triangular.dos_mp : multi-precision version suitable for integration

    References
    ----------
    .. [kogan2021] Kogan, E. and Gumbs, G. (2021) Green’s Functions and DOS for
       Some 2D Lattices. Graphene, 10, 1-12.
       https://doi.org/10.4236/graphene.2021.101001.

    Examples
    --------
    >>> eps = np.linspace(-1.5, 1.5, num=1000)
    >>> dos = gt.lattice.triangular.dos(eps, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.axvline(-4/9, color='black', linewidth=0.8)
    >>> _ = plt.axvline(0, color='black', linewidth=0.8)
    >>> _ = plt.plot(eps, dos)
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"DOS * $D$")
    >>> _ = plt.ylim(bottom=0)
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.show()

    """
    # FIXME: DOS/Gf at lower band-edge is somewhat undetermined
    D = half_bandwidth * 4 / 9
    eps = np.asarray(1.0 / D * eps)
    dos = np.zeros_like(eps)
    # implementation differs slightly from [kogan2021], as evaluating `ellipk`
    # is inaccurate around the singularity
    region1 = (-1.5 <= eps) & (eps <= -1)
    rr = np.sqrt(2*eps[region1] + 3)
    denom = (rr + 1)**3 * (3 - rr)
    numer = (rr - 1)**3 * (3 + rr)
    dos[region1] = 2 / np.sqrt(denom) * ellipkm1(-numer/denom)
    region2 = (-1 <= eps) & (eps <= +3)
    rr = np.sqrt(2*eps[region2] + 3)
    numer = (rr - 1)**3 * (3 + rr)
    dos[region2] = 0.5 / np.sqrt(rr) * ellipkm1(1/16*numer/rr)
    return 2 / np.pi**2 / D * dos


# ∫dϵ ϵ^m DOS(ϵ) for half-bandwidth D=1
# from: integral of dos_mp with mp.workdps(100)
# for m in range(0, 21, 1):
#     with mp.workdps(100, normalize_output=True):
#         res = mp.quad(lambda eps:  eps**m * dos_mp(eps), [-2/3, -4/9, +4/3])
#     print(res)
# rational numbers obtained by mp.identify
dos_moment_coefficients = {
    1: 0,
    2: 8/27,
    3: 32/243,
    4: 160/729,
    5: 0.19509221155311685,
    6: 0.24567167380762861,
    7: 0.26975713202406278,
    8: 0.32595653452907584,
    9: 0.38409863242932391,
    10: 0.46646891718728872,
    11: 0.5662391742471257,
    12: 0.69580884826902741,
    13: 0.85849121900290751,
    14: 1.06625635837817,
    15: 1.32983322599435,
    16: 1.66594704229184,
    17: 2.09437852592774,
    18: 2.64177488421009,
    19: 3.34185798350861,
    20: 4.23865856734991,
}


def dos_moment(m, half_bandwidth):
    """Calculate the `m` th moment of the triangular DOS.

    The moments are defined as :math:`∫dϵ ϵ^m DOS(ϵ)`.

    Parameters
    ----------
    m : int
        The order of the moment.
    half_bandwidth : float
        Half-bandwidth of the DOS of the 2D triangular lattice.

    Returns
    -------
    dos_moment : float
        The `m` th moment of the 2D triangular DOS.

    Raises
    ------
    NotImplementedError
        Currently only implemented for a few specific moments `m`.

    See Also
    --------
    gftool.lattice.triangular.dos

    """
    try:
        return dos_moment_coefficients[m] * half_bandwidth**m
    except KeyError as keyerr:
        raise NotImplementedError('Calculation of arbitrary moments not implemented.') from keyerr


def dos_mp(eps, half_bandwidth=1):
    r"""Multi-precision DOS of non-interacting 2D triangular lattice.

    The DOS diverges at `-4/9*half_bandwidth`.

    This function is particularity suited to calculate integrals of the form
    :math:`∫dϵ DOS(ϵ)f(ϵ)`. If you have problems with the convergence,
    consider using :math:`∫dϵ DOS(ϵ)[f(ϵ)-f(-4/9)] + f(-4/9)` to avoid the
    singularity.

    Parameters
    ----------
    eps : mpmath.mpf or mpf_like
        DOS is evaluated at points `eps`.
    half_bandwidth : mpmath.mpf or mpf_like
        Half-bandwidth of the DOS, DOS(`eps` < -2/3`half_bandwidth`) = 0,
        DOS(4/3`half_bandwidth` < `eps`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping
        :math:`t=4D/9`.

    Returns
    -------
    dos_mp : mpmath.mpf
        The value of the DOS.

    See Also
    --------
    gftool.lattice.triangular.dos : vectorized version suitable for array evaluations

    References
    ----------
    .. [kogan2021] Kogan, E. and Gumbs, G. (2021) Green’s Functions and DOS for
       Some 2D Lattices. Graphene, 10, 1-12.
       https://doi.org/10.4236/graphene.2021.101001.

    Examples
    --------
    Calculate integrals:

    >>> from mpmath import mp
    >>> mp.quad(gt.lattice.triangular.dos_mp, [-2/3, -4/9, 4/3])
    mpf('1.0')

    >>> eps = np.linspace(-2/3 - 0.1, 4/3 + 0.1, num=1000)
    >>> dos_mp = [gt.lattice.triangular.dos_mp(ee, half_bandwidth=1) for ee in eps]
    >>> dos_mp = np.array(dos_mp, dtype=np.float64)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.axvline(-4/9, color='black', linewidth=0.8)
    >>> _ = plt.axvline(0, color='black', linewidth=0.8)
    >>> _ = plt.plot(eps, dos_mp)
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"DOS * $D$")
    >>> _ = plt.ylim(bottom=0)
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.show()

    """
    D = mp.mpf(half_bandwidth) * mp.mpf('4/9')
    eps = mp.mpf(eps) / D
    if eps < mp.mpf('-1.5') or eps > +3:
        return mp.mpf('0')
    # higher precision around singularity is needed
    with mp.workdps(mp.dps*3.5, normalize_output=True):
        if mp.mpf('-1.5') <= eps <= -1:
            rr = mp.sqrt(2*eps + 3)
            z0 = (rr + 1)**3 * (3 - rr) / 4
            z1 = 4 * rr
            dos_ = 1 / mp.sqrt(z0) * mp.ellipk(z1/z0)
        elif -1 <= eps <= +3:
            rr = mp.sqrt(2*eps + 3)
            z0 = 4 * rr
            z1 = (rr + 1)**3 * (3 - rr) / 4
            dos_ = 1 / mp.sqrt(z0) * mp.ellipk(z1/z0)
        return 2 / np.pi**2 / D * dos_
