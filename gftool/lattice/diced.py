"""2D diced lattice.

The diced lattice can be decomposed into `~gftool.lattice.honeycomb` and a
dispersionless flat band.

:half_bandwidth: The half-bandwidth `D` corresponds to a nearest neighbor hopping
                 of `t=2**.5 * D / 3`

"""
from gftool.lattice import honeycomb


def gf_z(z, half_bandwidth):
    r"""
    Local Green's function of the 2D diced lattice.

    The Green's function of the 2D diced lattice can be expressed in terms
    of the 2D honeycomb lattice `gftool.lattice.honeycomb.gf_z`, and a
    non-dispersive peak.
    Compare the formulas in [varma2013]_, [horiguchi1974]_ to [horiguchi1972]_.

    The Green's function has singularities at `z=±half_bandwidth/3` and `z=0`.

    Parameters
    ----------
    z : complex np.ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : float
        Half-bandwidth of the DOS of the diced lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping
        :math:`t=2**.5 * D / 3`.

    Returns
    -------
    complex np.ndarray or complex
        Value of the diced lattice Green's function.

    See Also
    --------
    gftool.lattice.honeycomb.gf_z
    gftool.lattice.triangular.gf_z

    References
    ----------
    .. [varma2013] Varma, V. K. and Monien, H., 2013.
       Lattice Green's functions for kagome, diced, and hyperkagome lattices.
       Phys. Rev. E 87, 032109. https://doi.org/10.1103/PhysRevE.87.032109
    .. [horiguchi1974] Horiguchi, T., and Chen, C. C. (1974).
       Lattice Green's function for the diced lattice.
       J. Math. Phys. 15, 659. https://doi.org/10.1063/1.1666703
    .. [horiguchi1972] Horiguchi, T., 1972. Lattice Green’s Functions for the
       Triangular and Honeycomb Lattices. Journal of Mathematical Physics 13,
       1411–1419. https://doi.org/10.1063/1.1666155

    Examples
    --------
    >>> ww = np.linspace(-1.5, 1.5, num=1001, dtype=complex) + 1e-4j
    >>> gf_ww = gt.lattice.diced.gf_z(ww, half_bandwidth=1)

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
    peak = 1 / 3 / z
    return peak + 2/3*honeycomb.gf_z(z, half_bandwidth=D)


def dos(eps, half_bandwidth):
    r"""
    DOS of non-interacting 2D diced lattice.

    The delta-peak at `eps=0` is **ommited** and must be treated seperately!
    Without it, the DOS integrates to `2/3`.

    Besides the delta-peak, the DOS diverges at `eps=±half_bandwidth/3`.

    The Green's function and therefore the DOS of the 2D diced lattice can
    be expressed in terms of the 2D triangular lattice
    `gftool.lattice.triangular.dos`, see [varma2013]_. It is the same as (`2/3`) the
    DOS of the honeycomb lattice `gftool.lattice.honeycomb.dos`, plus a delta
    peak at `eps=0`, see [varma2013]_, [horiguchi1974], [horiguchi1972]_.

    Parameters
    ----------
    eps : float np.ndarray or float
        DOS is evaluated at points `eps`.
    half_bandwidth : float
        Half-bandwidth of the DOS, DOS(| `eps` | > `half_bandwidth`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping
        :math:`t=2**.5 * D / 3`.

    Returns
    -------
    float np.ndarray or float
        The value of the DOS.

    See Also
    --------
    gftool.lattice.diced.dos_mp : Multi-precision version suitable for integration.
    gftool.lattice.honeycomb.dos
    gftool.lattice.triangular.dos

    References
    ----------
    .. [varma2013] Varma, V. K. and Monien, H., 2013.
       Lattice Green's functions for kagome, diced, and hyperkagome lattices.
       Phys. Rev. E 87, 032109. https://doi.org/10.1103/PhysRevE.87.032109
    .. [horiguchi1974] Horiguchi, T., and Chen, C. C. (1974).
       Lattice Green's function for the diced lattice.
       J. Math. Phys. 15, 659. https://doi.org/10.1063/1.1666703
    .. [horiguchi1972] Horiguchi, T., 1972. Lattice Green’s Functions for the
       Triangular and Honeycomb Lattices. Journal of Mathematical Physics 13,
       1411–1419. https://doi.org/10.1063/1.1666155

    Examples
    --------
    >>> eps = np.linspace(-1.5, 1.5, num=501)
    >>> dos = gt.lattice.diced.dos(eps, half_bandwidth=1)

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
    return 2 / 3 * honeycomb.dos(eps, half_bandwidth)


def hilbert_transform(xi, half_bandwidth):
    r"""
    Hilbert transform of non-interacting DOS of the diced lattice.

    The Hilbert transform is defined

    .. math:: \tilde{D}(ξ) = ∫_{-∞}^{∞}dϵ \frac{DOS(ϵ)}{ξ − ϵ}

    The lattice Hilbert transform is the same as the non-interacting Green's
    function.

    Parameters
    ----------
    xi : complex np.ndarray or complex
        Point at which the Hilbert transform is evaluated.
    half_bandwidth : float
        Half-bandwidth of the DOS of the 2D diced lattice.

    Returns
    -------
    complex np.ndarray or complex
        Hilbert transform of `xi`.

    See Also
    --------
    gftool.lattice.diced.gf_z

    Notes
    -----
    Relation between nearest neighbor hopping `t` and half-bandwidth `D`

    .. math:: 3t/\sqrt{2} = D
    """
    return gf_z(xi, half_bandwidth)


dos_moment_coefficients = {m: 2*mom/3 for m, mom in honeycomb.dos_moment_coefficients.items()}


def dos_moment(m, half_bandwidth):
    """
    Calculate the `m` th moment of the diced DOS.

    The moments are defined as :math:`∫dϵ ϵ^m DOS(ϵ)`.

    Parameters
    ----------
    m : int
        The order of the moment.
    half_bandwidth : float
        Half-bandwidth of the DOS of the 2D diced lattice.

    Returns
    -------
    float
        The `m` th moment of the 2D diced DOS.

    Raises
    ------
    NotImplementedError
        Currently only implemented for a few specific moments `m`.

    See Also
    --------
    gftool.lattice.diced.dos
    """
    try:
        return dos_moment_coefficients[m] * half_bandwidth**m
    except KeyError as keyerr:
        raise NotImplementedError('Calculation of arbitrary moments not implemented.') from keyerr


def dos_mp(eps, half_bandwidth=1):
    r"""
    Multi-precision DOS of non-interacting 2D diced lattice.

    The delta-peak at `eps=0` is **ommited** and must be treated seperately!
    Without it, the DOS integrates to `2/3`.

    Besides the delta-peak, the DOS diverges at `eps=±half_bandwidth/3`.

    The Green's function and therefore the DOS of the 2D diced lattice can
    be expressed in terms of the 2D triangular lattice
    `gftool.lattice.triangular.dos`, see [varma2013]_. It is the same as (`2/3`) the
    DOS of the honeycomb lattice `gftool.lattice.honeycomb.dos`, plus a delta
    peak at `eps=0`, see [varma2013]_, [horiguchi1974], [horiguchi1972]_.

    Parameters
    ----------
    eps : mpmath.mpf or mpf_like
        DOS is evaluated at points `eps`.
    half_bandwidth : mpmath.mpf or mpf_like
        Half-bandwidth of the DOS, DOS(| `eps` | > `half_bandwidth`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping
        :math:`t=2**.5 * D / 3`.

    Returns
    -------
    float np.ndarray or float
        The value of the DOS.

    See Also
    --------
    gftool.lattice.diced.dos : Vectorized version suitable for array evaluations.
    gftool.lattice.honeycomb.dos_mp
    gftool.lattice.triangular.dos_mp

    References
    ----------
    .. [varma2013] Varma, V. K. and Monien, H., 2013.
       Lattice Green's functions for kagome, diced, and hyperkagome lattices.
       Phys. Rev. E 87, 032109. https://doi.org/10.1103/PhysRevE.87.032109
    .. [horiguchi1974] Horiguchi, T., and Chen, C. C. (1974).
       Lattice Green's function for the diced lattice.
       J. Math. Phys. 15, 659. https://doi.org/10.1063/1.1666703
    .. [horiguchi1972] Horiguchi, T., 1972. Lattice Green’s Functions for the
       Triangular and Honeycomb Lattices. Journal of Mathematical Physics 13,
       1411–1419. https://doi.org/10.1063/1.1666155

    Examples
    --------
    Calculated integrals

    >>> from mpmath import mp
    >>> mp.identify(mp.quad(gt.lattice.diced.dos_mp, [-1, -1/3, 0, +1/3, +1]))
    '(2/3)'

    >>> eps = np.linspace(-1.5, 1.5, num=501)
    >>> dos_mp = [gt.lattice.diced.dos(ee, half_bandwidth=1) for ee in eps]

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
    return 2 / 3 * honeycomb.dos_mp(eps, half_bandwidth)
