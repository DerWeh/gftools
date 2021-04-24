r"""2D honeycomb lattice.

The honeycomb lattice can be decomposed into `~gftool.lattice.triangular`
sublattices.

:half_bandwidth: The half-bandwidth `D` corresponds to a nearest neighbor hopping
                 of `t=2D/3`.

"""
from mpmath import mp

from gftool.lattice import triangular


def gf_z(z, half_bandwidth):
    r"""Local Green's function of the 2D honeycomb lattice.

    The Green's function of the 2D honeycomb lattice can be expressed in terms
    of the 2D triangular lattice `gftool.lattice.triangular.gf_z`,
    see [horiguchi1972]_.

    The Green's function has singularities at `z=±half_bandwidth/3`.

    Parameters
    ----------
    z : complex np.ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : float
        Half-bandwidth of the DOS of the honeycomb lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping
        :math:`t=2D/3`.

    Returns
    -------
    gf_z : complex np.ndarray or complex
        Value of the honeycomb lattice Green's function

    See Also
    --------
    gftool.lattice.triangular.gf_z

    References
    ----------
    .. [horiguchi1972] Horiguchi, T., 1972. Lattice Green’s Functions for the
       Triangular and Honeycomb Lattices. Journal of Mathematical Physics 13,
       1411–1419. https://doi.org/10.1063/1.1666155

    Examples
    --------
    >>> ww = np.linspace(-1.5, 1.5, num=501, dtype=complex) + 1e-64j
    >>> gf_ww = gt.lattice.honeycomb.gf_z(ww, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.axhline(0, color='black', linewidth=0.8)
    >>> _ = plt.plot(ww.real, gf_ww.real, label=r"$\Re G$")
    >>> _ = plt.plot(ww.real, gf_ww.imag, '--', label=r"$\Im G$")
    >>> _ = plt.ylabel(r"$G*D$")
    >>> _ = plt.xlabel(r"$\omega/D$")
    >>> _ = plt.xlim(left=ww.real.min(), right=ww.real.max())
    >>> _ = plt.legend()
    >>> plt.show()

    """
    D = half_bandwidth / 1.5
    z_rel = z / D
    return 2 / D * z_rel * triangular.gf_z(2*z_rel**2 - 1.5, half_bandwidth=9/4)


def dos(eps, half_bandwidth):
    r"""DOS of non-interacting 2D honeycomb lattice.

    The DOS diverges at `eps=±half_bandwidth/3`.
    The Green's function and therefore the DOS of the 2D honeycomb lattice can
    be expressed in terms of the 2D triangular lattice
    `gftool.lattice.triangular.dos`, see [horiguchi1972]_.

    Parameters
    ----------
    eps : float np.ndarray or float
        DOS is evaluated at points `eps`.
    half_bandwidth : float
        Half-bandwidth of the DOS, DOS(| `eps` | > `half_bandwidth`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping
        :math:`t=2D/3`.

    Returns
    -------
    dos : float np.ndarray or float
        The value of the DOS.

    See Also
    --------
    gftool.lattice.honeycomb.dos_mp : multi-precision version suitable for integration
    gftool.lattice.triangular.dos

    References
    ----------
    .. [horiguchi1972] Horiguchi, T., 1972. Lattice Green’s Functions for the
       Triangular and Honeycomb Lattices. Journal of Mathematical Physics 13,
       1411–1419. https://doi.org/10.1063/1.1666155

    Examples
    --------
    >>> eps = np.linspace(-1.5, 1.5, num=501)
    >>> dos = gt.lattice.honeycomb.dos(eps, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> for pos in (-1/3, 0, +1/3):
    ...     _ = plt.axvline(pos, color='black', linewidth=0.8)
    >>> _ = plt.plot(eps, dos)
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"DOS * $D$")
    >>> _ = plt.ylim(bottom=0)
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.show()

    """
    D = half_bandwidth / 1.5
    eps_rel = eps / D
    return 2 / D * abs(eps_rel) * triangular.dos(2*eps_rel**2 - 1.5, half_bandwidth=9/4)


def hilbert_transform(xi, half_bandwidth):
    r"""Hilbert transform of non-interacting DOS of the honeycomb lattice.

    The Hilbert transform is defined

    .. math:: \tilde{D}(ξ) = ∫_{-∞}^{∞}dϵ \frac{DOS(ϵ)}{ξ − ϵ}

    The lattice Hilbert transform is the same as the non-interacting Green's
    function.

    Parameters
    ----------
    xi : complex np.ndarray or complex
        Point at which the Hilbert transform is evaluated
    half_bandwidth : float
        half-bandwidth of the DOS of the 2D honeycomb lattice

    Returns
    -------
    hilbert_transform : complex ndarray or complex
        Hilbert transform of `xi`.

    Notes
    -----
    Relation between nearest neighbor hopping `t` and half-bandwidth `D`

    .. math:: 3t/2 = D

    See Also
    --------
    gftool.lattice.honeycomb.gf_z

    """
    return gf_z(xi, half_bandwidth)


# ∫dϵ ϵ^m DOS(ϵ) for half-bandwidth D=1
# from: integral of dos_mp with mp.workdps(100)
# for m in range(0, 22, 2):
#     with mp.workdps(100):
#         print(mp.quad(lambda eps: 2 * eps**m * dos_mp(eps), [0, 1/3, 1])
# rational numbers obtained by mp.identify
dos_moment_coefficients = {
    2: 1/3,
    4: 5/27,
    6: 31/243,
    8: 71/729,
    10: 0.0787989635726261,
    12: 0.0661766781260761,
    14: 0.0570430207680627,
    16: 0.0501259782365305,
    18: 0.0447055266609815,
    20: 0.0403432070418971,
}


def dos_moment(m, half_bandwidth):
    """Calculate the `m` th moment of the honeycomb DOS.

    The moments are defined as :math:`∫dϵ ϵ^m DOS(ϵ)`.

    Parameters
    ----------
    m : int
        The order of the moment.
    half_bandwidth : float
        Half-bandwidth of the DOS of the 2D honeycomb lattice.

    Returns
    -------
    dos_moment : float
        The `m` th moment of the 2D honeycomb DOS.

    Raises
    ------
    NotImplementedError
        Currently only implemented for a few specific moments `m`.

    See Also
    --------
    gftool.lattice.honeycomb.dos

    """
    if m % 2:  # odd moments vanish due to symmetry
        return 0
    try:
        return dos_moment_coefficients[m] * half_bandwidth**m
    except KeyError as keyerr:
        raise NotImplementedError('Calculation of arbitrary moments not implemented.') from keyerr


def dos_mp(eps, half_bandwidth=1):
    r"""Multi-precision DOS of non-interacting 2D honeycomb lattice.

    The DOS diverges at `eps=±half_bandwidth/3`.

    This function is particularity suited to calculate integrals of the form
    :math:`∫dϵ DOS(ϵ)f(ϵ)`. If you have problems with the convergence,
    consider removing singularities, e.g. split the integral

    .. math::
       ∫^0 dϵ DOS(ϵ)[f(ϵ) - f(-D/3)] + ∫_0 dϵ DOS(ϵ)[f(ϵ) - f(+D/3)] + [f(-D/3) + f(+D/3)]/2

    where :math:`D` is the `half_bandwidth`, or symmetrize the integral.

    The Green's function and therefore the DOS of the 2D honeycomb lattice can
    be expressed in terms of the 2D triangular lattice
    `gftool.lattice.triangular.dos_mp`, see [horiguchi1972]_.

    Parameters
    ----------
    eps : mpmath.mpf or mpf_like
        DOS is evaluated at points `eps`.
    half_bandwidth : mpmath.mpf or mpf_like
        Half-bandwidth of the DOS, DOS(| `eps` | > `half_bandwidth`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping
        :math:`t=2D/3`.

    Returns
    -------
    dos_mp : mpmath.mpf
        The value of the DOS.

    See Also
    --------
    gftool.lattice.honeycomb.dos : vectorized version suitable for array evaluations
    gftool.lattice.triangular.dos_mp

    References
    ----------
    .. [horiguchi1972] Horiguchi, T., 1972. Lattice Green’s Functions for the
       Triangular and Honeycomb Lattices. Journal of Mathematical Physics 13,
       1411–1419. https://doi.org/10.1063/1.1666155

    Examples
    --------
    Calculated integrals

    >>> from mpmath import mp
    >>> mp.quad(gt.lattice.honeycomb.dos_mp, [-1, -1/3, 0, +1/3, +1])
    mpf('1.0')

    >>> eps = np.linspace(-1.5, 1.5, num=501)
    >>> dos_mp = [gt.lattice.honeycomb.dos_mp(ee, half_bandwidth=1) for ee in eps]

    >>> import matplotlib.pyplot as plt
    >>> for pos in (-1/3, 0, +1/3):
    ...     _ = plt.axvline(pos, color='black', linewidth=0.8)
    >>> _ = plt.plot(eps, dos_mp)
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"DOS * $D$")
    >>> _ = plt.ylim(bottom=0)
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.show()

    """
    D_inv = mp.mpf('1.5') / mp.mpf(half_bandwidth)
    eps_rel = mp.mpf(eps) * D_inv
    t_dos = triangular.dos_mp(2*eps_rel**2 - mp.mpf('1.5'), half_bandwidth=mp.mpf('9/4'))
    return 2 * D_inv * mp.fabs(eps_rel) * t_dos
