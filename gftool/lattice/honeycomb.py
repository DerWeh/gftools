r"""2D honeycomb lattice.

:half_bandwidth: The half-bandwidth `D` corresponds to a nearest neighbor hopping
                 of `t=2D/3`.

"""
from gftool.lattice import triangular


def gf_z(z, half_bandwidth):
    r"""Local Green's function of the 2D honeycomb lattice.

    The Green's function of the 2D honeycomb lattice can be expressed in terms
    of the 2D triangular lattice `gftool.lattice.triangular.gf_z`,
    see [horiguchi1972]_.

    The Green's function has singularities at `z=±1/3`

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
    >>> ww = np.linspace(-1.5, 1.5, num=500, dtype=complex) + 1e-64j
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
        :math:`t=4D/9`.

    Returns
    -------
    dos : float np.ndarray or float
        The value of the DOS.

    See Also
    --------
    gftool.lattice.triangular.dos

    References
    ----------
    .. [horiguchi1972] Horiguchi, T., 1972. Lattice Green’s Functions for the
       Triangular and Honeycomb Lattices. Journal of Mathematical Physics 13,
       1411–1419. https://doi.org/10.1063/1.1666155

    Examples
    --------
    >>> eps = np.linspace(-1.5, 1.5, num=500)
    >>> dos = gt.lattice.honeycomb.dos(eps, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.plot(eps, dos)
    >>> _ = plt.axvline(-2/3, color='black', linewidth=0.8)
    >>> _ = plt.axvline(+4/3, color='black', linewidth=0.8)
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"DOS * $D$")
    >>> _ = plt.axvline(0, color='black', linewidth=0.8)
    >>> _ = plt.ylim(bottom=0)
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.show()

    """
    D = half_bandwidth / 1.5
    eps_rel = eps / D
    return 2 / D * abs(eps_rel) * triangular.dos(2*eps_rel**2 - 1.5, half_bandwidth=9/4)
