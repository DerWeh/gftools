"""Bethe lattice with infinite coordination number.

This is in fact no real lattice, but a tree. It corresponds to a semi-circular
DOS.

:half_bandwidth: The half_bandwidth corresponds to a scaled nearest neighbor
                 hopping of `t=D/2`

"""
import numpy as np

from gftool.precision import PRECISE_TYPES as _PRECISE_TYPES


def gf_z(z, half_bandwidth):
    r"""Local Green's function of Bethe lattice for infinite coordination number.

    .. math::
        G(z) = 2*(z - s\sqrt{z^2 - D^2})/D^2

    where :math:`D` is the half bandwidth and :math:`s=sgn[ℑ{ξ}]`. See
    [georges1996]_.

    Parameters
    ----------
    z : complex ndarray or complex
        Green's function is evaluated at complex frequency `z`
    half_bandwidth : float
        Half-bandwidth of the DOS of the Bethe lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`

    Returns
    -------
    gf_z : complex ndarray or complex
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
    >>> _ = plt.axhline(0, color='black', linewidth=0.8)
    >>> _ = plt.axvline(0, color='black', linewidth=0.8)
    >>> _ = plt.xlim(left=ww.min(), right=ww.max())
    >>> _ = plt.legend()
    >>> plt.grid()
    >>> plt.show()

    """
    z_rel = np.array(z / half_bandwidth, dtype=np.complex256)
    try:
        complex_pres = np.complex256 if z.dtype in _PRECISE_TYPES else np.complex
    except AttributeError:
        complex_pres = np.complex
    gf_z = 2./half_bandwidth*z_rel*(1 - np.sqrt(1 - z_rel**-2))
    return gf_z.astype(dtype=complex_pres, copy=False)


def gf_d1_z(z, half_bandwidth):
    """First derivative of local Green's function of Bethe lattice for infinite coordination number.

    Parameters
    ----------
    z : complex ndarray or complex
        Green's function is evaluated at complex frequency `z`
    half_bandwidth : float
        half-bandwidth of the DOS of the Bethe lattice
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`

    Returns
    -------
    gf_d1_z : complex ndarray or complex
        Value of the derivative of the Green's function

    See Also
    --------
    gftool.lattice.bethe.gf_z

    """
    z_rel_inv = np.array(half_bandwidth / z, dtype=np.complex256)
    try:
        complex_pres = np.complex256 if z.dtype in _PRECISE_TYPES else np.complex
    except AttributeError:
        complex_pres = np.complex
    sqrt = np.sqrt(1 - z_rel_inv**2)
    gf_d1 = 2. / half_bandwidth**2 * (1 - 1/sqrt)
    return gf_d1.astype(dtype=complex_pres, copy=False)


def gf_d2_z(z, half_bandwidth):
    """Second derivative of local Green's function of Bethe lattice for infinite coordination number.

    Parameters
    ----------
    z : complex ndarray or complex
        Green's function is evaluated at complex frequency `z`
    half_bandwidth : float
        half-bandwidth of the DOS of the Bethe lattice
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`

    Returns
    -------
    gf_d2_z : complex ndarray or complex
        Value of the Green's function

    See Also
    --------
    gftool.lattice.bethe.gf_z

    """
    z_rel = np.array(z / half_bandwidth, dtype=np.complex256)
    try:
        complex_pres = np.complex256 if z.dtype in _PRECISE_TYPES else np.complex
    except AttributeError:
        complex_pres = np.complex
    sqrt = np.sqrt(1 - z_rel**-2)
    gf_d2 = 2. / half_bandwidth**3 * z_rel * sqrt / (1 - z_rel**2)**2
    return gf_d2.astype(dtype=complex_pres, copy=False)


def hilbert_transform(xi, half_bandwidth):
    r"""Hilbert transform of non-interacting DOS of the Bethe lattice.

    The Hilbert transform is defined

    .. math::
        \tilde{D}(ξ) = ∫_{-∞}^{∞}dϵ \frac{DOS(ϵ)}{ξ − ϵ}

    The lattice Hilbert transform is the same as the non-interacting Green's
    function.

    Parameters
    ----------
    xi : complex ndarray or complex
        Point at which the Hilbert transform is evaluated
    half_bandwidth : float
        half-bandwidth of the DOS of the Bethe lattice

    Returns
    -------
    hilbert_transform : complex ndarray or complex
        Hilbert transform of `xi`.

    Notes
    -----
    Relation between nearest neighbor hopping `t` and half-bandwidth `D`

    .. math::
        2t = D

    See Also
    --------
    gftool.lattice.bethe.gf_z

    """
    return gf_z(xi, half_bandwidth)


def dos(eps, half_bandwidth):
    r"""DOS of non-interacting Bethe lattice for infinite coordination number.

    Parameters
    ----------
    eps : float ndarray or float
        DOS is evaluated at points `eps`.
    half_bandwidth : float
        Half-bandwidth of the DOS, DOS(| `eps` | > `half_bandwidth`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`

    Returns
    -------
    dos : float ndarray or float
        The value of the DOS.

    Examples
    --------
    >>> eps = np.linspace(-1.1, 1.1, num=500)
    >>> dos = gt.lattice.bethe.dos(eps, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.plot(eps, dos)
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"DOS * $D$")
    >>> _ = plt.axhline(0, color='black', linewidth=0.8)
    >>> _ = plt.axvline(0, color='black', linewidth=0.8)
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.grid()
    >>> plt.show()

    """
    eps_rel = np.asarray(eps / half_bandwidth)
    dos = np.zeros_like(eps_rel)
    nonzero = (abs(eps_rel) < 1) | np.iscomplex(eps)
    dos[nonzero] = 2. / (np.pi*half_bandwidth) * np.sqrt(1 - eps_rel[nonzero]**2)
    return dos


# ∫dϵ ϵ^m DOS(ϵ) for half-bandwidth D=1
# from:
# https://www.wolframalpha.com/input/?i=integrate+sqrt(1+-+x%5E2)%2F(.5*pi)+*x%5E%7B2*n%7Ddx+from+-1+to+1+assuming+n+is+integer
dos_moment_coefficients = {
    2: 0.25,
    4: 0.125,
    6: 0.078125,
    8: 0.0546875,
    # not sure if results form wolframalpha are accurate enough, calculated them with mpmath
    # 10: 0.0410156,
    # 12: 0.0322266,
    # 14: 0.0261841,
    # 16: 0.0218201,
    # 18: 0.0185471,
    # 20: 0.0160179,
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
