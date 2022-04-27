"""Green's function corresponding to a box DOS.

This doesn't correspond to any real lattice. It is mostly meant as very simple
test case, for which we have analytic expressions.

"""
import numpy as np

_SMALL = np.finfo(np.float64).eps**0.25


def gf_z(z, half_bandwidth):
    r"""Local Green's function corresponding to a box DOS.

    .. math:: G(z) = \ln(\frac{z + D}{z - D}) / 2D

    Parameters
    ----------
    z : complex array_like or complex
        Green's function is evaluated at complex frequency `z`
    half_bandwidth : float
        Half-bandwidth of the box DOS.

    Returns
    -------
    gf_z : complex np.ndarray or complex
        Value of the Green's function corresponding to a box DOS.

    Examples
    --------
    >>> ww = np.linspace(-1.5, 1.5, num=500)
    >>> gf_ww = gt.lattice.box.gf_z(ww, half_bandwidth=1)

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
    z_rel = z / half_bandwidth
    return 0.5 / half_bandwidth * np.emath.log((z_rel + 1) / (z_rel - 1))


def dos(eps, half_bandwidth):
    r"""Box-shaped DOS.

    Parameters
    ----------
    eps : float array_like or float
        DOS is evaluated at points `eps`.
    half_bandwidth : float
        Half-bandwidth of the DOS, DOS(| `eps` | > `half_bandwidth`) = 0.

    Returns
    -------
    dos : float np.ndarray or float
        The value of the DOS.

    Examples
    --------
    >>> eps = np.linspace(-1.1, 1.1, num=500)
    >>> dos = gt.lattice.box.dos(eps, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.plot(eps, dos)
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"DOS * $D$")
    >>> _ = plt.axvline(0, color='black', linewidth=0.8)
    >>> _ = plt.ylim(bottom=0)
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.show()

    """
    dos = np.where(abs(eps) < half_bandwidth, 0.5/half_bandwidth, 0)
    return dos


def dos_moment(m, half_bandwidth):
    """Calculate the `m` th moment of the box DOS.

    The moments are defined as :math:`∫dϵ ϵ^m DOS(ϵ)`.

    Parameters
    ----------
    m : int
        The order of the moment.
    half_bandwidth : float
        Half-bandwidth of the DOS of the box lattice.

    Returns
    -------
    dos_moment : float
        The `m` th moment of the box DOS.

    See Also
    --------
    gftool.lattice.box.dos

    """
    if m % 2:  # odd moments vanish due to symmetry
        return 0
    return half_bandwidth**m / (m + 1)


def gf_ret_t(tt, half_bandwidth, center=0):
    r"""Local retarded-time local Green's function corresponding to a box DOS.

    .. math:: G(t) = -1j Θ(t) \sin(Dt)/Dt

    where :math:`D` is the half bandwidth.

    Parameters
    ----------
    tt : float array_like or float
        Green's function is evaluated at time `tt`.
    half_bandwidth : float
        Half-bandwidth of the box DOS.
    center : float
        Position of the center of the box DOS.
        This parameter is **not** given in units of `half_bandwidth`.

    Returns
    -------
    gf_ret_t : complex np.ndarray or complex
        Value of the retarded-time Green's function corresponding to a box DOS.

    Examples
    --------
    >>> tt = np.linspace(0, 50, 1500)
    >>> gf_tt = gt.lattice.box.gf_ret_t(tt, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.axhline(0, color='black', linewidth=0.8)
    >>> _ = plt.plot(tt, gf_tt.imag, label=r"$\Im G$")
    >>> _ = plt.plot(tt, gf_tt.real, '--', label=r"$\Re G$")
    >>> _ = plt.xlabel(r"$t*D$")
    >>> _ = plt.ylabel(r"$G$")
    >>> _ = plt.xlim(left=tt.min(), right=tt.max())
    >>> _ = plt.legend()
    >>> plt.show()

    """
    tt = half_bandwidth*tt
    gf = np.zeros_like(tt, dtype=complex)
    retard = tt.real >= 0
    small = retard & (abs(tt) < _SMALL)
    # Taylor expansion for small tt, to avoid 1/tt
    tt2 = tt[small]**2
    gf[small] = -1j*(1 - 1/6*tt2 + 1/120*tt2**2)
    big = retard & ~small
    gf[big] = -1j * np.sin(tt)[big] / tt[big]
    if center:
        return gf*np.exp(-1j*center*tt)
    return gf
