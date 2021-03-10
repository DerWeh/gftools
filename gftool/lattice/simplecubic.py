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

    Implements equations (1.24 - 1.26) from [delves2001].

    Parameters
    ----------
    z : mp.mpf or mpf_like
        Energy point at which the DOS is evaluated.
        Will be converted to a multi-precision float `mp.mpf`.
    half_bandwidth : mp.mpf or mpf_like
        Half-bandwidth of the DOS.

    Returns
    -------
    gf_z : mp.mpf
        Value of the Green's function at complex energy 'z'.

    Examples
    --------
    >>> eps = np.linspace(-1.1, 1.1, num=101) + 1j*1e-7
    >>> with mp.workdps(15):
    >>>     gf_ww = [gt.lattice.simplecubic.gf_z_mp(ee)[0] for ee in eps]

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.plot(eps, -gf_ww.imag, 'x-')
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"-Im(G) * $D$")
    >>> _ = plt.axvline(1/3, color="black", linestyle="--")
    >>> _ = plt.axvline(-1/3, color="black", linestyle="--")
    >>> _ = plt.axvline(0, color='black', linewidth=0.8)
    >>> _ = plt.ylim(bottom=0)
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.show()

    References
    ----------
    .. [economou2006] Economou, E. N. Green's Functions in Quantum Physics.
       Springer, 2006.

    .. [delves2001] Delves, R. T. and Joyce, G. S., Ann. Phys. 291, 71 (2001).

    """
    z = 3 * mp.mpc(z) / half_bandwidth
    z_sqr = 1 / z**2
    xi = mp.sqrt(1 - mp.sqrt(1 - z_sqr)) / mp.sqrt(1 + mp.sqrt(1 - 9*z_sqr))
    k2 = 16 * xi**3 / ((1 - xi)**3 * (1 + 3*xi))
    green = (1 - 9*xi**4) * (2 * mp.ellipk(k2) / mp.pi)**2 / ((1 - xi)**3 * (1 + 3*xi)) / z

    return 3 * green / half_bandwidth
