"""2D Kagome lattice.

The DOS is finite in the interval :math:`[-2D/3, 4D/3]`, where :math:`D` is the
half-bandwidth.

The kagome lattice can be decomposed into `~gftool.lattice.triangular` and a
dispersionless flat band. The dispersive part looks like the
`~gftool.lattice.honeycomb` lattice.

:half_bandwidth: The half-bandwidth `D` corresponds to a nearest neighbor hopping
                 of `t=2D/3`

"""
from mpmath import mp

from gftool.lattice import honeycomb


def gf_z(z, half_bandwidth):
    r"""Local Green's function of the 2D kagome lattice.

    The Green's function of the 2D kagome lattice can be expressed in terms
    of the 2D triangular lattice `gftool.lattice.triangular.gf_z`, and a
    non-dispersive peak, see [kogan2021]_.
    Omitting the non-dispersive peak, it corresponds to
    `gftool.lattice.honeycomb.gf_z` shifted by `half_bandwidth/3`.

    The Green's function has singularities for `z/half_bandwidth in [-2/3, 0, 2/3]`.

    Parameters
    ----------
    z : complex np.ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : float
        Half-bandwidth of the DOS of the kagome lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping
        :math:`t=2D/3`.

    Returns
    -------
    gf_z : complex np.ndarray or complex
        Value of the kagome lattice Green's function

    See Also
    --------
    gftool.lattice.triangular.gf_z

    References
    ----------
    .. [varm2013] Varma, V.K., Monien, H., 2013. Lattice Green’s functions for
       kagome, diced, and hyperkagome lattices. Phys. Rev. E 87, 032109.
       https://doi.org/10.1103/PhysRevE.87.032109
    .. [kogan2021] Kogan, E., Gumbs, G., 2020. Green’s Functions and DOS for
       Some 2D Lattices. Graphene 10, 1–12.
       https://doi.org/10.4236/graphene.2021.101001

    Examples
    --------
    >>> ww = np.linspace(-1.5, 1.5, num=1001, dtype=complex) + 1e-4j
    >>> gf_ww = gt.lattice.kagome.gf_z(ww, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.axhline(0, color='black', linewidth=0.8)
    >>> _ = plt.plot(ww.real, gf_ww.real, label=r"$\Re G$")
    >>> _ = plt.plot(ww.real, gf_ww.imag, '--', label=r"$\Im G$")
    >>> _ = plt.ylabel(r"$G*D$")
    >>> _ = plt.xlabel(r"$\omega/D$")
    >>> _ = plt.xlim(left=ww.real.min(), right=ww.real.max())
    >>> _ = plt.ylim(bottom=-5, top=5)
    >>> _ = plt.legend()
    >>> plt.show()

    """
    D = half_bandwidth
    peak = 1 / (3*z + 2*D)
    return peak + 2/3*honeycomb.gf_z(z - D/3, half_bandwidth=D)


def dos(eps, half_bandwidth):
    r"""DOS of non-interacting 2D kagome lattice.

    The delta-peak at `eps=-2*half_bandwidth/3` is **ommited** and must be
    treated seperately! Without it, the DOS integrates to `2/3`.

    Besides the delta-peak, the DOS diverges at `eps=0` and `eps=2*half_bandwidth/3`.

    The Green's function and therefore the DOS of the 2D kagome lattice can
    be expressed in terms of the 2D triangular lattice
    `gftool.lattice.triangular.dos`, see [kogan2021]_.
    Omitting the non-dispersive peak, it corresponds to
    `gftool.lattice.honeycomb.dos` shifted by `half_bandwidth/3`.

    Parameters
    ----------
    eps : float np.ndarray or float
        DOS is evaluated at points `eps`.
    half_bandwidth : float
        Half-bandwidth of the DOS, DOS(`eps` < -2/3`half_bandwidth`) = 0,
        DOS(4/3`half_bandwidth` < `eps`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping
        :math:`t=2D/3`.

    Returns
    -------
    dos : float np.ndarray or float
        The value of the DOS.

    See Also
    --------
    gftool.lattice.kagome.dos_mp : multi-precision version suitable for integration
    gftool.lattice.triangular.dos
    gftool.lattice.honeycomb.dos

    References
    ----------
    .. [varm2013] Varma, V.K., Monien, H., 2013. Lattice Green’s functions for
       kagome, diced, and hyperkagome lattices. Phys. Rev. E 87, 032109.
       https://doi.org/10.1103/PhysRevE.87.032109
    .. [kogan2021] Kogan, E., Gumbs, G., 2020. Green’s Functions and DOS for
       Some 2D Lattices. Graphene 10, 1–12.
       https://doi.org/10.4236/graphene.2021.101001

    Examples
    --------
    >>> eps = np.linspace(-1.5, 1.5, num=1001)
    >>> dos = gt.lattice.kagome.dos(eps, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> for pos in (-2/3, 0, +2/3):
    ...     _ = plt.axvline(pos, color='black', linewidth=0.8)
    >>> _ = plt.plot(eps, dos)
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"DOS * $D$")
    >>> _ = plt.ylim(bottom=0)
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.show()

    """
    D = half_bandwidth
    return 2 / 3 * honeycomb.dos(eps - D/3, half_bandwidth=D)


def hilbert_transform(xi, half_bandwidth):
    r"""Hilbert transform of non-interacting DOS of the kagome lattice.

    The Hilbert transform is defined

    .. math:: \tilde{D}(ξ) = ∫_{-∞}^{∞}dϵ \frac{DOS(ϵ)}{ξ − ϵ}

    The lattice Hilbert transform is the same as the non-interacting Green's
    function.

    Parameters
    ----------
    xi : complex np.ndarray or complex
        Point at which the Hilbert transform is evaluated
    half_bandwidth : float
        half-bandwidth of the DOS of the 2D kagome lattice

    Returns
    -------
    hilbert_transform : complex np.ndarray or complex
        Hilbert transform of `xi`.

    Notes
    -----
    Relation between nearest neighbor hopping `t` and half-bandwidth `D`

    .. math:: 3t/2 = D

    See Also
    --------
    gftool.lattice.kagome.gf_z

    """
    return gf_z(xi, half_bandwidth)


# ∫dϵ ϵ^m DOS(ϵ) for half-bandwidth D=1
# from: integral of dos_mp with mp.workdps(100)
# for m in range(0, 21, 1):
#     with mp.workdps(100, normalize_output=True):
#         res = mp.quad(lambda eps:  eps**m * dos_mp(eps), [-2/3, 0, +1/3, +2/3, +4/3])
#         res += (-2 / 3)**m / 3  # add delta peak
#     print(res)
# rational numbers obtained by mp.identify
dos_moment_coefficients = {
    0: 1,
    1: 0,
    2: 4/9,
    3: 4/27,
    4: 28/81,
    5: 20/81,
    6: 88/243,
    7: 0.358481938728852,
    8: 0.457857033988721,
    9: 0.518416907991668,
    10: 0.640552761266067,
    11: 0.766504654326632,
    12: 0.946076798741534,
    13: 1.16111979818393,
    14: 1.44297318255669,
    15: 1.79568687705621,
    16: 2.248655455081,
    17: 2.82371565536896,
    18: 3.55975081121742,
    19: 4.49985828877178,
    20: 5.70448609391951,
}


def dos_moment(m, half_bandwidth):
    """Calculate the `m` th moment of the kagome DOS.

    The moments are defined as :math:`∫dϵ ϵ^m DOS(ϵ)`.

    Parameters
    ----------
    m : int
        The order of the moment.
    half_bandwidth : float
        Half-bandwidth of the DOS of the 2D kagome lattice.

    Returns
    -------
    dos_moment : float
        The `m` th moment of the 2D kagome DOS.

    Raises
    ------
    NotImplementedError
        Currently only implemented for a few specific moments `m`.

    See Also
    --------
    gftool.lattice.kagome.dos

    """
    try:
        return dos_moment_coefficients[m] * half_bandwidth**m
    except KeyError as keyerr:
        raise NotImplementedError('Calculation of arbitrary moments not implemented.') from keyerr


def dos_mp(eps, half_bandwidth=1):
    r"""Multi-precision DOS of non-interacting 2D kagome lattice.

    The delta-peak at `eps=-2*half_bandwidth/3` is **ommited** and must be
    treated seperately! Without it, the DOS integrates to `2/3`.

    Besides the delta-peak, the DOS diverges at `eps=0` and `eps=2*half_bandwidth/3`.

    Parameters
    ----------
    eps : mpmath.mpf or mpf_like
        DOS is evaluated at points `eps`.
    half_bandwidth : mpmath.mpf or mpf_like
        Half-bandwidth of the DOS, DOS(`eps` < -2/3`half_bandwidth`) = 0,
        DOS(4/3`half_bandwidth` < `eps`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping
        :math:`t=2D/3`.

    Returns
    -------
    dos : float np.ndarray or float
        The value of the DOS.

    See Also
    --------
    gftool.lattice.kagome.dos_mp : vectorized version suitable for array evaluations
    gftool.lattice.triangular.dos_mp
    gftool.lattice.honeycomb.dos_mp

    References
    ----------
    .. [varm2013] Varma, V.K., Monien, H., 2013. Lattice Green’s functions for
       kagome, diced, and hyperkagome lattices. Phys. Rev. E 87, 032109.
       https://doi.org/10.1103/PhysRevE.87.032109
    .. [kogan2021] Kogan, E., Gumbs, G., 2020. Green’s Functions and DOS for
       Some 2D Lattices. Graphene 10, 1–12.
       https://doi.org/10.4236/graphene.2021.101001

    Examples
    --------
    Calculated integrals

    >>> from mpmath import mp
    >>> mp.identify(mp.quad(gt.lattice.kagome.dos_mp, [-2/3, 0, 1/3, 2/3, 4/3]))
    '(2/3)'

    >>> eps = np.linspace(-1.5, 1.5, num=1001)
    >>> dos_mp = [gt.lattice.kagome.dos(ee, half_bandwidth=1) for ee in eps]

    >>> import matplotlib.pyplot as plt
    >>> for pos in (-2/3, 0, +2/3):
    ...     _ = plt.axvline(pos, color='black', linewidth=0.8)
    >>> _ = plt.plot(eps, dos_mp)
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"DOS * $D$")
    >>> _ = plt.ylim(bottom=0)
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.show()

    """
    D = mp.mpf(half_bandwidth)
    return mp.mpf("2/3") * honeycomb.dos_mp(eps - D/3, half_bandwidth=D)
