"""Bethe lattice with infinite coordination number.

This is in fact no real lattice, but a tree. It corresponds to a semi-circular
DOS.

:half_bandwidth: The half_bandwidth corresponds to a scaled nearest neighbor
                 hopping of `t=D/2`

"""
import numpy as np

from mpmath import mp

from gftool.precision import PRECISE_TYPES as _PRECISE_TYPES


def gf_z(z, half_bandwidth):
    r"""Local Green's function of Bethe lattice for infinite coordination number.

    .. math:: G(z) = 2(z - s\sqrt{z^2 - D^2})/D^2

    where :math:`D` is the half bandwidth and :math:`s=sgn[ℑ{ξ}]`. See
    [georges1996]_.

    Parameters
    ----------
    z : complex array_like or complex
        Green's function is evaluated at complex frequency `z`
    half_bandwidth : float
        Half-bandwidth of the DOS of the Bethe lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`

    Returns
    -------
    gf_z : complex np.ndarray or complex
        Value of the Bethe Green's function

    References
    ----------
    .. [georges1996] Georges et al., Rev. Mod. Phys. 68, 13 (1996)
       https://doi.org/10.1103/RevModPhys.68.13

    Examples
    --------
    >>> ww = np.linspace(-1.5, 1.5, num=500)
    >>> gf_ww = gt.lattice.bethe.gf_z(ww, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.plot(ww, gf_ww.real, label=r"$\Re G$")
    >>> _ = plt.plot(ww, gf_ww.imag, '--', label=r"$\Im G$")
    >>> _ = plt.xlabel(r"$\omega/D$")
    >>> _ = plt.ylabel(r"$G*D$")
    >>> _ = plt.axhline(0, color='black', linewidth=0.8)
    >>> _ = plt.xlim(left=ww.min(), right=ww.max())
    >>> _ = plt.legend()
    >>> plt.show()

    """
    z_rel = np.array(z / half_bandwidth, dtype=np.complex256)
    try:
        complex_pres = np.complex256 if z.dtype in _PRECISE_TYPES else complex
    except AttributeError:
        complex_pres = complex
    gf_z = 2./half_bandwidth*z_rel*(1 - np.sqrt(1 - z_rel**-2))
    return gf_z.astype(dtype=complex_pres, copy=False)


def gf_d1_z(z, half_bandwidth):
    """First derivative of local Green's function of Bethe lattice for infinite coordination number.

    Parameters
    ----------
    z : complex array_like or complex
        Green's function is evaluated at complex frequency `z`
    half_bandwidth : float
        half-bandwidth of the DOS of the Bethe lattice
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`

    Returns
    -------
    gf_d1_z : complex np.ndarray or complex
        Value of the derivative of the Green's function

    See Also
    --------
    gftool.lattice.bethe.gf_z

    """
    z_rel_inv = np.array(half_bandwidth / z, dtype=np.complex256)
    try:
        complex_pres = np.complex256 if z.dtype in _PRECISE_TYPES else complex
    except AttributeError:
        complex_pres = complex
    sqrt = np.sqrt(1 - z_rel_inv**2)
    gf_d1 = 2. / half_bandwidth**2 * (1 - 1/sqrt)
    return gf_d1.astype(dtype=complex_pres, copy=False)


def gf_d2_z(z, half_bandwidth):
    """Second derivative of local Green's function of Bethe lattice for infinite coordination number.

    Parameters
    ----------
    z : complex array_like or complex
        Green's function is evaluated at complex frequency `z`
    half_bandwidth : float
        half-bandwidth of the DOS of the Bethe lattice
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`

    Returns
    -------
    gf_d2_z : complex np.ndarray or complex
        Value of the Green's function

    See Also
    --------
    gftool.lattice.bethe.gf_z

    """
    z_rel = np.array(z / half_bandwidth, dtype=np.complex256)
    try:
        complex_pres = np.complex256 if z.dtype in _PRECISE_TYPES else complex
    except AttributeError:
        complex_pres = complex
    sqrt = np.sqrt(1 - z_rel**-2)
    gf_d2 = 2. / half_bandwidth**3 * z_rel * sqrt / (1 - z_rel**2)**2
    return gf_d2.astype(dtype=complex_pres, copy=False)


def gf_z_inv(gf, half_bandwidth):
    r"""Inverse of local Green's function of Bethe lattice for infinite coordination number.

    .. math:: R(G) = (D/2)^2 G + 1/G

    where :math:`R(z) = G^{-1}(z)` is the inverse of the Green's function.


    Parameters
    ----------
    gf : complex array_like or complex
        Value of the local Green's function.
    half_bandwidth : float
        Half-bandwidth of the DOS of the Bethe lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`

    Returns
    -------
    z : complex np.ndarray or complex
        The inverse of the Bethe Green's function `gf_z(gf_z_inv(g, D), D)=g`.

    See Also
    --------
    gftool.lattice.bethe.gf_z

    References
    ----------
    .. [georges1996] Georges et al., Rev. Mod. Phys. 68, 13 (1996)
       https://doi.org/10.1103/RevModPhys.68.13

    Examples
    --------
    >>> ww = np.linspace(-1.5, 1.5, num=500) + 1e-4j
    >>> gf_ww = gt.lattice.bethe.gf_z(ww, half_bandwidth=1)
    >>> np.allclose(ww, gt.lattice.bethe.gf_z_inv(gf_ww, half_bandwidth=1))
    True

    """
    return (0.5 * half_bandwidth)**2 * gf + 1./gf


def hilbert_transform(xi, half_bandwidth):
    r"""Hilbert transform of non-interacting DOS of the Bethe lattice.

    The Hilbert transform is defined as:

    .. math:: \tilde{D}(ξ) = ∫_{-∞}^{∞}dϵ \frac{DOS(ϵ)}{ξ − ϵ}

    The lattice Hilbert transform is the same as the non-interacting Green's
    function.

    Parameters
    ----------
    xi : complex array_like or complex
        Point at which the Hilbert transform is evaluated
    half_bandwidth : float
        half-bandwidth of the DOS of the Bethe lattice

    Returns
    -------
    hilbert_transform : complex np.ndarray or complex
        Hilbert transform of `xi`.

    Notes
    -----
    Relation between nearest neighbor hopping `t` and half-bandwidth `D`:

    .. math:: 2t = D

    See Also
    --------
    gftool.lattice.bethe.gf_z

    """
    return gf_z(xi, half_bandwidth)


def dos(eps, half_bandwidth):
    r"""DOS of non-interacting Bethe lattice for infinite coordination number.

    Parameters
    ----------
    eps : float array_like or float
        DOS is evaluated at points `eps`.
    half_bandwidth : float
        Half-bandwidth of the DOS, DOS(| `eps` | > `half_bandwidth`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`

    Returns
    -------
    dos : float np.ndarray or float
        The value of the DOS.

    See Also
    --------
    gftool.lattice.bethe.dos_mp : multi-precision version suitable for integration

    References
    ----------
    .. [economou2006] Economou, E. N. Green's Functions in Quantum Physics.
       Springer, 2006.

    Examples
    --------
    >>> eps = np.linspace(-1.1, 1.1, num=500)
    >>> dos = gt.lattice.bethe.dos(eps, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.plot(eps, dos)
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"DOS * $D$")
    >>> _ = plt.axvline(0, color='black', linewidth=0.8)
    >>> _ = plt.ylim(bottom=0)
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.show()

    """
    eps_rel = np.asarray(eps / half_bandwidth)
    dos = np.zeros_like(eps_rel)
    nonzero = (abs(eps_rel) < 1) | np.iscomplex(eps)
    dos[nonzero] = 2. / (np.pi*half_bandwidth) * np.sqrt(1 - eps_rel[nonzero]**2)
    return dos


# ∫dϵ ϵ^m DOS(ϵ) for half-bandwidth D=1
# from: integral of dos_mp with mp.workdps(100)
# for m in range(0, 22, 2):
#     with mp.workdps(100):
#         print(mp.quad(lambda eps: 2 * eps**m * dos_mp(eps), [0, 1]))
# rational numbers obtained by mp.identify
dos_moment_coefficients = {
    2: 0.25,
    4: 0.125,
    6: 5/64,
    8: 7/128,
    10: 21/512,
    12: 0.0322265625,
    14: 0.02618408203125,
    16: 0.021820068359375,
    18: 0.01854705810546875,
    20: 0.016017913818359375,
}


def dos_moment(m, half_bandwidth):
    """Calculate the `m` th moment of the Bethe DOS.

    The moments are defined as :math:`∫dϵ ϵ^m DOS(ϵ)`.

    Parameters
    ----------
    m : int
        The order of the moment.
    half_bandwidth : float
        Half-bandwidth of the DOS of the Bethe lattice.

    Returns
    -------
    dos_moment : float
        The `m` th moment of the Bethe DOS.

    Raises
    ------
    NotImplementedError
        Currently only implemented for a few specific moments `m`.

    See Also
    --------
    gftool.lattice.bethe.dos

    """
    if m % 2:  # odd moments vanish due to symmetry
        return 0
    try:
        return dos_moment_coefficients[m] * half_bandwidth**m
    except KeyError as keyerr:
        raise NotImplementedError('Calculation of arbitrary moments not implemented.') from keyerr


def dos_mp(eps, half_bandwidth=1):
    r"""Multi-precision DOS of non-interacting Bethe lattice for infinite coordination number.

    This function is particularly suited to calculate integrals of the form
    :math:`∫dϵ DOS(ϵ)f(ϵ)`.

    Parameters
    ----------
    eps : mpmath.mpf or mpf_like
        DOS is evaluated at points `eps`.
    half_bandwidth : mpmath.mpf or mpf_like
        Half-bandwidth of the DOS, DOS(| `eps` | > `half_bandwidth`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`

    Returns
    -------
    dos_mp : mpmath.mpf
        The value of the DOS.

    See Also
    --------
    gftool.lattice.bethe.dos : vectorized version suitable for array evaluations

    References
    ----------
    .. [economou2006] Economou, E. N. Green's Functions in Quantum Physics.
       Springer, 2006.

    Examples
    --------
    Calculate integrals:

    >>> from mpmath import mp
    >>> mp.quad(gt.lattice.bethe.dos_mp, [-1, 1])
    mpf('1.0')

    >>> eps = np.linspace(-1.1, 1.1, num=500)
    >>> dos_mp = [gt.lattice.bethe.dos_mp(ee, half_bandwidth=1) for ee in eps]
    >>> dos_mp = np.array(dos_mp, dtype=np.float64)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.plot(eps, dos_mp)
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"DOS * $D$")
    >>> _ = plt.axvline(0, color='black', linewidth=0.8)
    >>> _ = plt.ylim(bottom=0)
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.show()

    """
    eps, half_bandwidth = mp.mpf(eps), mp.mpf(half_bandwidth)
    if mp.fabs(eps) > half_bandwidth:
        return mp.mpf('0')
    return 2 / (mp.pi * half_bandwidth) * mp.sqrt(-mp.powm1(eps / half_bandwidth, mp.mpf('2')))
