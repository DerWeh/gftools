r"""2D Lieb lattice.

The lieb lattice can be decomposed into `~gftool.lattice.square` and a
dispersionless flat band.

:half_bandwidth: The half-bandwidth `D` corresponds to a nearest neighbor hopping
                 of `t=D * 2**1.5`

"""
from mpmath import mp

from gftool.lattice import square


def gf_z(z, half_bandwidth):
    r"""Local Green's function of the 2D Lieb lattice.

    The Green's function of the 2D Lieb lattice can be expressed in terms
    of the 2D square lattice `gftool.lattice.square.gf_z`, and a non-dispersive
    peak, see [kogan2021]_.

    The Green's function has singularities for
    `z/half_bandwidth in [-2**-0.5, 0, 2**-0.5]`.

    Parameters
    ----------
    z : complex np.ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : float
        Half-bandwidth of the DOS of the Lieb lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D * 2**1.5`.

    Returns
    -------
    gf_z : complex np.ndarray or complex
        Value of the Lieb lattice Green's function

    See Also
    --------
    gftool.lattice.square.gf_z

    References
    ----------
    .. [kogan2021] Kogan, E., Gumbs, G., 2020. Green’s Functions and DOS for
       Some 2D Lattices. Graphene 10, 1–12.
       https://doi.org/10.4236/graphene.2021.101001

    Examples
    --------
    >>> ww = np.linspace(-1.5, 1.5, num=1001, dtype=complex) + 1e-4j
    >>> gf_ww = gt.lattice.lieb.gf_z(ww, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.axhline(0, color='black', linewidth=0.8)
    >>> _ = plt.plot(ww.real, gf_ww.real, label=r"$\Re G$")
    >>> _ = plt.plot(ww.real, gf_ww.imag, '--', label=r"$\Im G$")
    >>> _ = plt.ylabel(r"$G*D$")
    >>> _ = plt.xlabel(r"$\omega/D$")
    >>> _ = plt.xlim(left=ww.real.min(), right=ww.real.max())
    >>> _ = plt.ylim(bottom=-5.0, top=5.0)
    >>> _ = plt.legend()
    >>> plt.show()

    """
    D = half_bandwidth * 2**-1.5
    z_rel = z / D
    peak = (1 / 3) / z
    return peak + 2/(3*D)*z_rel*square.gf_z(z_rel**2 - 4, half_bandwidth=4)


def dos(eps, half_bandwidth):
    r"""DOS of non-interacting 2D Lieb lattice.

    The delta-peak at `eps=0` is **ommited** and must be treated seperately!
    Without it, the DOS integrates to `2/3`.

    Besides the delta-peak, the DOS diverges at `eps=±half_bandwidth/2**0.5`.

    The Green's function and therefore the DOS of the 2D Lieb lattice can
    be expressed in terms of the 2D square lattice `gftool.lattice.square.dos`,
    see [kogan2021]_.

    Parameters
    ----------
    eps : float np.ndarray or float
        DOS is evaluated at points `eps`.
    half_bandwidth : float
        Half-bandwidth of the DOS, DOS(| `eps` | > `half_bandwidth`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D * 2**1.5`

    Returns
    -------
    dos : float np.ndarray or float
        The value of the DOS.

    See Also
    --------
    gftool.lattice.lieb.dos_mp : multi-precision version suitable for integration
    gftool.lattice.square.dos

    References
    ----------
    .. [kogan2021] Kogan, E., Gumbs, G., 2020. Green’s Functions and DOS for
       Some 2D Lattices. Graphene 10, 1–12.
       https://doi.org/10.4236/graphene.2021.101001

    Examples
    --------
    >>> eps = np.linspace(-1.5, 1.5, num=1001)
    >>> dos = gt.lattice.lieb.dos(eps, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> for pos in (-2**-0.5, 0, +2**-0.5):
    ...     _ = plt.axvline(pos, color='black', linewidth=0.8)
    >>> _ = plt.plot(eps, dos)
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"DOS * $D$")
    >>> _ = plt.ylim(bottom=0)
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.show()

    """
    D = half_bandwidth
    eps_rel = eps / D
    return 4 / (3 * D) * abs(eps_rel) * square.dos(2*eps_rel**2 - 1, half_bandwidth=1)


def hilbert_transform(xi, half_bandwidth):
    r"""Hilbert transform of non-interacting DOS of the lieb lattice.

    The Hilbert transform is defined

    .. math:: \tilde{D}(ξ) = ∫_{-∞}^{∞}dϵ \frac{DOS(ϵ)}{ξ − ϵ}

    The lattice Hilbert transform is the same as the non-interacting Green's
    function.

    Parameters
    ----------
    xi : complex np.ndarray or complex
        Point at which the Hilbert transform is evaluated
    half_bandwidth : float
        half-bandwidth of the DOS of the 2D lieb lattice

    Returns
    -------
    hilbert_transform : complex np.ndarray or complex
        Hilbert transform of `xi`.

    Notes
    -----
    Relation between nearest neighbor hopping `t` and half-bandwidth `D`

    .. math:: t = D 2^{3/2}

    See Also
    --------
    gftool.lattice.lieb.gf_z

    """
    return gf_z(xi, half_bandwidth)


# ∫dϵ ϵ^m DOS(ϵ) for half-bandwidth D=1, the δ-peak does not contribute for m>0
# from: integral of dos_mp with mp.workdps(100)
# for m in range(2, 21, 2):
#     with mp.workdps(100, normalize_output=True):
#         res = 2*mp.quad(lambda eps:  eps**m * dos_mp(eps), [0, 2**0.5, 1])
#     print(res)
# rational numbers obtained by mp.identify
dos_moment_coefficients = {
    0: 1,
    2: 1/3,
    4: 5/24,
    6: 7/48,
    8: 0.110026041666667,
    10: 0.0875651041666667,
    12: 0.0724690755208333,
    14: 0.0617472330729167,
    16: 0.0537835756937663,
    18: 0.0476494630177816,
    20: 0.0427826742331187,
}


def dos_moment(m, half_bandwidth):
    """Calculate the `m` th moment of the Lieb DOS.

    The moments are defined as :math:`∫dϵ ϵ^m DOS(ϵ)`.

    Parameters
    ----------
    m : int
        The order of the moment.
    half_bandwidth : float
        Half-bandwidth of the DOS of the 2D Lieb lattice.

    Returns
    -------
    dos_moment : float
        The `m` th moment of the 2D Lieb DOS.

    Raises
    ------
    NotImplementedError
        Currently only implemented for a few specific moments `m`.

    See Also
    --------
    gftool.lattice.lieb.dos

    """
    if m % 2:  # odd moments vanish due to symmetry
        return 0
    try:
        return dos_moment_coefficients[m] * half_bandwidth**m
    except KeyError as keyerr:
        raise NotImplementedError('Calculation of arbitrary moments not implemented.') from keyerr


def dos_mp(eps, half_bandwidth=1):
    r"""Multi-precision DOS of non-interacting 2D lieb lattice.

    The delta-peak at `eps=0` is **ommited** and must be treated seperately!
    Without it, the DOS integrates to `2/3`.

    Besides the delta-peak, the DOS diverges at `eps=±half_bandwidth/2**0.5`.

    This function is particularity suited to calculate integrals of the form
    :math:`∫dϵ DOS(ϵ)f(ϵ)`. If you have problems with the convergence,
    consider removing singularities, e.g. split the integral

    .. math::
       ∫^0 dϵ DOS(ϵ)[f(ϵ) - f(-D/\sqrt{2})] + ∫_0 dϵ DOS(ϵ)[f(ϵ) - f(+D/\sqrt{3})] \\
       + [f(-D/\sqrt{2}) + f(0) + f(+D/\sqrt{2})]/3

    where :math:`D` is the `half_bandwidth`, or symmetrize the integral.

    The Green's function and therefore the DOS of the 2D Lieb lattice can
    be expressed in terms of the 2D square lattice `gftool.lattice.square.dos`,
    see [kogan2021]_.

    Parameters
    ----------
    eps : mpmath.mpf or mpf_like
        DOS is evaluated at points `eps`.
    half_bandwidth : mpmath.mpf or mpf_like
        Half-bandwidth of the DOS, DOS(| `eps` | > `half_bandwidth`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D * 2**1.5`

    Returns
    -------
    dos_mp : mpmath.mpf
        The value of the DOS.

    See Also
    --------
    gftool.lattice.lieb.dos : vectorized version suitable for array evaluations
    gftool.lattice.square.dos_mp

    References
    ----------
    .. [kogan2021] Kogan, E., Gumbs, G., 2020. Green’s Functions and DOS for
       Some 2D Lattices. Graphene 10, 1–12.
       https://doi.org/10.4236/graphene.2021.101001

    Examples
    --------
    Calculated integrals

    >>> from mpmath import mp
    >>> mp.identify(mp.quad(gt.lattice.lieb.dos_mp, [-1, -2**-0.5, 0, 2**-0.5, 1]))
    '(2/3)'

    >>> eps = np.linspace(-1.5, 1.5, num=501)
    >>> dos_mp = [gt.lattice.lieb.dos_mp(ee, half_bandwidth=1) for ee in eps]

    >>> import matplotlib.pyplot as plt
    >>> for pos in (-2**-0.5, 0, +2**-0.5):
    ...     _ = plt.axvline(pos, color='black', linewidth=0.8)
    >>> _ = plt.plot(eps, dos_mp)
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"DOS * $D$")
    >>> _ = plt.ylim(bottom=0)
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.show()

    """
    D_inv = mp.mpf('2')**mp.mpf('1.5') / mp.mpf(half_bandwidth)
    eps_rel = mp.mpf(eps) * D_inv
    s_dos = square.dos_mp(eps_rel**mp.mpf('2') - mp.mpf('4'), half_bandwidth=mp.mpf('4'))
    return mp.mpf('2/3') * D_inv * mp.fabs(eps_rel) * s_dos
