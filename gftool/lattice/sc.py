r"""3D simple cubic (sc) lattice.

The dispersion of the 3D simple cubic lattice is given by

.. math:: ϵ_{k_x, k_y, k_z} = 2t [\cos(k_x) + \cos(k_y) + \cos(k_z)]

which takes values in :math:`ϵ_{k_x, k_y, k_z} ∈ [-6t, +6t] = [-D, +D]`.

:half_bandwidth: The half_bandwidth corresponds to a nearest neighbor hopping
                 of `t=D/6`

"""
import numpy as np

from numpy.lib.scimath import sqrt
from mpmath import mp

from gftool._util import _u_ellipk


def gf_z(z, half_bandwidth=1):
    r"""Local Green's function of 3D simple cubic lattice.

    Has a van Hove singularity (continuous but not differentiable) at
    `z = ±D/3`.

    Implements equations (1.24 - 1.26) from [delves2001]_.

    Parameters
    ----------
    z : complex np.ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : float
        Half-bandwidth of the DOS of the simple cubic lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping
        :math:`t=D/6`.

    Returns
    -------
    gf_z : complex np.ndarray or complex
        Value of the simple cubic Green's function at complex energy `z`.

    References
    ----------
    .. [economou2006] Economou, E. N. Green's Functions in Quantum Physics.
       Springer, 2006.
    .. [delves2001] Delves, R. T. and Joyce, G. S., Ann. Phys. 291, 71 (2001).
       https://doi.org/10.1006/aphy.2001.6148

    Examples
    --------
    >>> ww = np.linspace(-1.1, 1.1, num=500)
    >>> gf_ww = gt.lattice.sc.gf_z(ww)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.axhline(0, color="black", linewidth=0.8)
    >>> _ = plt.axvline(-1/3, color="black", linewidth=0.8)
    >>> _ = plt.axvline(+1/3, color="black", linewidth=0.8)
    >>> _ = plt.plot(ww.real, gf_ww.real, label=r"$\Re G$")
    >>> _ = plt.plot(ww.real, gf_ww.imag, label=r"$\Im G$")
    >>> _ = plt.ylabel(r"$G*D$")
    >>> _ = plt.xlabel(r"$\omega/D$")
    >>> _ = plt.xlim(left=ww.min(), right=ww.max())
    >>> _ = plt.legend()
    >>> plt.show()

    """
    D_inv = 3 / half_bandwidth
    z = D_inv * z
    z_sqr = z**-2
    xi = sqrt(1 - sqrt(1 - z_sqr)) / sqrt(1 + sqrt(1 - 9*z_sqr))
    denom_inv = 1 / ((1 - xi)**3 * (1 + 3*xi))
    k2 = 16 * xi**3 * denom_inv
    gf_z = (1 - 9*xi**4) * (2 / np.pi * _u_ellipk(k2))**2 * denom_inv / z
    return D_inv * gf_z


def hilbert_transform(xi, half_bandwidth=1):
    r"""Hilbert transform of non-interacting DOS of the simple cubic lattice.

    The Hilbert transform is defined

    .. math:: \tilde{D}(ξ) = ∫_{-∞}^{∞}dϵ \frac{DOS(ϵ)}{ξ − ϵ}

    The lattice Hilbert transform is the same as the non-interacting Green's
    function.

    Parameters
    ----------
    xi : complex np.ndarray or complex
        Point at which the Hilbert transform is evaluated
    half_bandwidth : float
        half-bandwidth of the DOS of the 3D simple cubic lattice

    Returns
    -------
    hilbert_transform : complex np.ndarray or complex
        Hilbert transform of `xi`.

    Notes
    -----
    Relation between nearest neighbor hopping `t` and half-bandwidth `D`

    .. math:: 6t = D

    See Also
    --------
    gftool.lattice.sc.gf_z

    """
    return gf_z(xi, half_bandwidth)


def dos(eps, half_bandwidth=1):
    r"""Local Green's function of 3D simple cubic lattice.

    Has a van Hove singularity (continuous but not differentiable) at
    `abs(eps) = D/3`.

    Implements Eq. 7.37 from [joyce1973]_ for the special case of `eps = 0`,
    otherwise identical to `-gf_z.imag/np.pi`.

    Parameters
    ----------
    eps : float np.ndarray or float
        DOS is evaluated at points `eps`.
    half_bandwidth : float
        Half-bandwidth of the DOS of the simple cubic lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping
        :math:`t=D/6`.

    Returns
    -------
    dos : float np.ndarray or float
        The value of the DOS.

    References
    ----------
    .. [economou2006] Economou, E. N. Green's Functions in Quantum Physics.
       Springer, 2006.
    .. [joyce1973] G. S. Joyce, Phil. Trans. of the Royal Society of London A,
       273, 583 (1973). https://www.jstor.org/stable/74037
    .. [katsura1971] S. Katsura et al., J. Math. Phys., 12, 895 (1971).
       https://doi.org/10.1063/1.1665663

    Examples
    --------
    >>> eps = np.linspace(-1.1, 1.1, num=501)
    >>> dos = gt.lattice.sc.dos(eps)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.axhline(0, color="black", linewidth=0.8)
    >>> _ = plt.axvline(-1/3, color="black", linewidth=0.8)
    >>> _ = plt.axvline(+1/3, color="black", linewidth=0.8)
    >>> _ = plt.plot(eps, dos)
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"DOS * $D$")
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.show()

    """
    D_inv = 3 / half_bandwidth
    eps = np.asarray(abs(D_inv * eps))
    dos_ = np.zeros_like(eps)
    finite = (eps > 0) & (eps < 3)  # 0 will be treated separately
    # Green's function but avoid (1 ± 1/eps**2) for small eps
    eps2 = eps[finite]**2
    xi = sqrt(eps[finite] - sqrt(eps2 - 1)) / sqrt(eps[finite] + sqrt(eps2 - 9))
    denom_inv = 1 / ((1 - xi)**3 * (1 + 3*xi))
    k2 = 16 * xi**3 * denom_inv
    gf_ = (1 - 9*xi**4) * (2 / np.pi * _u_ellipk(k2))**2 * denom_inv / eps[finite]
    dos_[finite] = -1. / np.pi * gf_.imag
    zero = eps == 0
    if np.any(zero):
        km2 = 0.25 * (2 - mp.sqrt(3))
        dos_[zero] = (2 / mp.pi**2) * mp.ellipk(km2) * mp.ellipk(1 - km2) / mp.pi
    return D_inv * dos_


# ∫dϵ ϵ^m DOS(ϵ) for half-bandwidth D=1
# from: integral of dos_mp with mp.workdps(50)
# `2*mp.quad(lambda eps: eps**4 * gt.lattice.sc.dos_mp(eps), [0, mp.mpf('1/3)])`
# rational numbers obtained by mp.identify
dos_moment_coefficients = {
    2: 1/6,
    4: 5/72,
    6: 0.039866255144032922,
    8: 0.026631087105624143,
    10: 0.01939193244170096,
    12: 0.014928527975706617,
    14: 0.011948953080810005,
    16: 0.0098437704453147492,
    18: 0.0082915600061680671,
    20: 0.0071083541490866967,
}


def dos_moment(m, half_bandwidth):
    """Calculate the `m` th moment of the simple cubic DOS.

    The moments are defined as :math:`∫dϵ ϵ^m DOS(ϵ)`.

    Parameters
    ----------
    m : int
        The order of the moment.
    half_bandwidth : float
        Half-bandwidth of the DOS of the 3D simple cubic lattice.

    Returns
    -------
    dos_moment : float
        The `m` th moment of the 3D simple cubic DOS.

    Raises
    ------
    NotImplementedError
        Currently only implemented for a few specific moments `m`.

    See Also
    --------
    gftool.lattice.sc.dos

    """
    if m % 2:  # odd moments vanish due to symmetry
        return 0
    try:
        return dos_moment_coefficients[m] * half_bandwidth**m
    except KeyError as keyerr:
        raise NotImplementedError('Calculation of arbitrary moments not implemented.') from keyerr


def gf_z_mp(z, half_bandwidth=1):
    r"""Multi-precision Green's function of non-interacting 3D simple cubic lattice.

    Has a van Hove singularity (continuous but not differentiable) at
    `z = ±D/3`.

    Implements equations (1.24 - 1.26) from [delves2001]_.

    Parameters
    ----------
    z : mpmath.mpc or mpc_like
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : mpmath.mpf or mpf_like
        Half-bandwidth of the DOS of the simple cubic lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping
        :math:`t=D/6`.

    Returns
    -------
    gf_z : mpmath.mpc
        Value of the Green's function at complex energy `z`.

    References
    ----------
    .. [economou2006] Economou, E. N. Green's Functions in Quantum Physics.
       Springer, 2006.
    .. [delves2001] Delves, R. T. and Joyce, G. S., Ann. Phys. 291, 71 (2001).
       https://doi.org/10.1006/aphy.2001.6148

    Examples
    --------
    >>> ww = np.linspace(-1.1, 1.1, num=500)
    >>> gf_ww = np.array([gt.lattice.sc.gf_z_mp(wi) for wi in ww])

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.axhline(0, color="black", linewidth=0.8)
    >>> _ = plt.axvline(-1/3, color="black", linewidth=0.8)
    >>> _ = plt.axvline(+1/3, color="black", linewidth=0.8)
    >>> _ = plt.plot(ww.real, gf_ww.astype(complex).real, label=r"$\Re G$")
    >>> _ = plt.plot(ww.real, gf_ww.astype(complex).imag, label=r"$\Im G$")
    >>> _ = plt.ylabel(r"$G*D$")
    >>> _ = plt.xlabel(r"$\omega/D$")
    >>> _ = plt.xlim(left=ww.min(), right=ww.max())
    >>> _ = plt.legend()
    >>> plt.show()

    """
    D_inv = 3 / half_bandwidth
    z = D_inv * mp.mpc(z)
    z_sqr = 1 / z**2
    xi = mp.sqrt(1 - mp.sqrt(1 - z_sqr)) / mp.sqrt(1 + mp.sqrt(1 - 9*z_sqr))
    k2 = 16 * xi**3 / ((1 - xi)**3 * (1 + 3*xi))
    green = (1 - 9*xi**4) * (2 * mp.ellipk(k2) / mp.pi)**2 / ((1 - xi)**3 * (1 + 3*xi)) / z

    return D_inv * green


def dos_mp(eps, half_bandwidth=1):
    r"""Multi-precision DOS of non-interacting 3D simple cubic lattice.

    Has a van Hove singularity (continuous but not differentiable) at
    `abs(eps) = D/3`.

    Implements Eq. 7.37 from [joyce1973]_ for the special case of `eps = 0`,
    otherwise calls `gf_z_mp`.

    Parameters
    ----------
    eps : mpmath.mpf or mpf_like
        DOS is evaluated at points `eps`.
    half_bandwidth : mpmath.mpf or mpf_like
        Half-bandwidth of the DOS of the simple cubic lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping
        :math:`t=D/6`.

    Returns
    -------
    dos_mp : mpmath.mpf
        The value of the DOS.

    References
    ----------
    .. [economou2006] Economou, E. N. Green's Functions in Quantum Physics.
       Springer, 2006.
    .. [joyce1973] G. S. Joyce, Phil. Trans. of the Royal Society of London A,
       273, 583 (1973). https://www.jstor.org/stable/74037
    .. [katsura1971] S. Katsura et al., J. Math. Phys., 12, 895 (1971).
       https://doi.org/10.1063/1.1665663


    Examples
    --------
    >>> eps = np.linspace(-1.1, 1.1, num=501)
    >>> dos_mp = [gt.lattice.sc.dos_mp(ee, half_bandwidth=1) for ee in eps]
    >>> dos_mp = np.array(dos_mp, dtype=np.float64)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.axvline(1/3, color="black", linewidth=0.8)
    >>> _ = plt.axvline(-1/3, color="black", linewidth=0.8)
    >>> _ = plt.plot(eps, dos_mp)
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"DOS * $D$")
    >>> _ = plt.axvline(0, color="black", linewidth=0.8)
    >>> _ = plt.ylim(bottom=0)
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.show()

    """
    D_inv = 3 / half_bandwidth
    eps = mp.fabs(eps)
    if eps == 0:
        km2 = 0.25 * (2 - mp.sqrt(3))
        return D_inv * (2 / mp.pi**2) * mp.ellipk(km2) * mp.ellipk(1 - km2) / mp.pi
    return -mp.im(gf_z_mp(eps, half_bandwidth)) / mp.pi
