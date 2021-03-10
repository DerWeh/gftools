"""3D simple cubic lattice.

:half_bandwidth: The half_bandwidth corresponds to a nearest neighbor hopping
                 of `t=D/6`

"""
from mpmath import mp


def dos_mp(eps, half_bandwidth=1):
    r"""Multi-precision DOS of non-interacting 3D simple cubic lattice.

    Has a van Hove singularity (continuous but not differentiable) at
    `abs(t) = D/3`.

    Implements equations (3.5 - 3.10) from [morita1970].

    Parameters
    ----------
    t : float ndarray or float
        DOS is evaluated at points `t`.

    Returns
    -------
    dos_mp : float ndarray or float
        The value of the DOS.

    Examples
    --------
    >>> eps = np.linspace(-1.1, 1.1, num=500)
    >>> dos = gt.lattice.simplecubic.dos(eps, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.plot(eps, dos_mp)
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"DOS * $D$")
    >>> _ = plt.axvline(1/3, color="black", linestyle="--")
    >>> _ = plt.axvline(0, color='black', linewidth=0.8)
    >>> _ = plt.ylim(bottom=0)
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.show()

    References
    ----------
    .. [economou2006] Economou, E. N. Green's Functions in Quantum Physics.
       Springer, 2006.

    .. [morita1970] Morita, T. and Horiguchi, T. J., Math. Phys. 12, 981 (1970).

    """
    eps = 3 * mp.fabs(eps) / half_bandwidth
    delta = mp.eps**(1/mp.mpf(2))
    integrand = lambda phi: mp.ellipk(1 - (eps - mp.cos(phi))**2 / 4)
    if eps > 1:
        endpoint = mp.acos(eps - 2)
        integ1 = mp.quad(integrand, [0, endpoint]) / mp.pi**2
        return mp.re( 3 * integ1 / half_bandwidth / mp.pi )
    singularity = mp.acos(eps)
    endpoint = mp.acos(eps - 2) if eps > 1 else mp.pi
    integ1 = mp.quad(integrand, [0, singularity-delta]) / mp.pi**2
    integ2 = mp.quad(integrand, [singularity+delta, endpoint]) / mp.pi**2
    return mp.re( 3 * (integ1+integ2) / half_bandwidth / mp.pi )


def gf_z_mp(z, half_bandwidth=1):
    r"""Multi-precision Green's function of non-interacting 3D simple cubic lattice.

    Has a van Hove singularity (continuous but not differentiable) at
    `abs(z) = D/3`.

    Implements equations (1.24 - 1.26) from [delves2001]_.

    Parameters
    ----------
    z : mpmath.mpc or mpc_like
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : mp.mpf or mpf_like
        Half-bandwidth of the DOS of the simple cubic lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping
        :math:`t=D/6`.

    Returns
    -------
    gf_z : mpmath.mpc
        Value of the Green's function at complex energy 'z'.

    References
    ----------
    .. [economou2006] Economou, E. N. Green's Functions in Quantum Physics.
       Springer, 2006.
    .. [delves2001] Delves, R. T. and Joyce, G. S., Ann. Phys. 291, 71 (2001).
       https://doi.org/10.1006/aphy.2001.6148

    Examples
    --------
    >>> ww = np.linspace(-1.1, 1.1, num=500)
    >>> gf_ww = np.array([gt.lattice.simplecubic.gf_z_mp(wi) for wi in ww])

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.axhline(0, color='black', linewidth=0.8)
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
    z = 3 * mp.mpc(z) / half_bandwidth
    z_sqr = 1 / z**2
    xi = mp.sqrt(1 - mp.sqrt(1 - z_sqr)) / mp.sqrt(1 + mp.sqrt(1 - 9*z_sqr))
    k2 = 16 * xi**3 / ((1 - xi)**3 * (1 + 3*xi))
    green = (1 - 9*xi**4) * (2 * mp.ellipk(k2) / mp.pi)**2 / ((1 - xi)**3 * (1 + 3*xi)) / z

    return 3 * green / half_bandwidth
