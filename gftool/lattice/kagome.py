"""2D Kagome lattice.

The DOS is finite in the interval :math:`[-2D/3, 4D/3]`, where :math:`D` is the
half-bandwidth.

The kagome lattice can be decomposed into `~gftool.lattice.triangular` and a
dispersionless flat band. The dispersive part looks like the
`~gftool.lattice.honeycomb` lattice.

:half_bandwidth: The half-bandwidth `D` corresponds to a nearest neighbor hopping
                 of `t=2D/3`

"""
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
