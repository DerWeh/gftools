r"""2D Lieb lattice.

The lieb lattice can be decomposed into `~gftool.lattice.square` and a
dispersionless flat band.

:half_bandwidth: The half-bandwidth `D` corresponds to a nearest neighbor hopping
                 of `t=D * 2**1.5`

"""
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
    D = half_bandwidth / (2 * 2**0.5)
    eps_rel = eps / D
    return 2 / (3 * D) * abs(eps_rel) * square.dos(eps_rel**2 - 4, half_bandwidth=4)
