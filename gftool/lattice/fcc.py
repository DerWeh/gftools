r"""3D face-centered cubic (fcc) lattice.

The dispersion of the 3D face-centered cubic lattice is given by

.. math::

   ϵ_{k_x,k_y,k_z} = 4t [\cos(k_x/2)\cos(k_y/2) + \cos(k_x/2)\cos(k_z/2) + \cos(k_y/2) \cos(k_z/2)]

which takes values in :math:`ϵ_{k_x, k_y, k_z} ∈ [-4t, +12t] = [-0.5D, +1.5D]`.

:half_bandwidth: The half_bandwidth corresponds to a nearest neighbor hopping
                 of `t=D/8`.

"""
import numpy as np

from mpmath import mp

from gftool._util import _u_ellipk


def _signed_sqrt(z):
    """Square root with correct sign for fcc lattice."""
    # sign = np.where((z.real < 0) & (z.imag < 0), -1, 1)
    sign = np.where(z.real < 0, -1, 1)
    factor = np.where(sign == 1, 1, -1j)
    return factor * np.lib.scimath.sqrt(sign*z)


def gf_z(z, half_bandwidth):
    r"""Local Green's function of the 3D face-centered cubic (fcc) lattice.

    Note, that the spectrum is asymmetric and in :math:`[-D/2, 3D/2]`,
    where :math:`D` is the half-bandwidth.

    Has a van Hove singularity at `z=-half_bandwidth/2` (divergence) and at
    `z=0` (continuous but not differentiable).

    Implements equations (2.16), (2.17) and (2.11) from [morita1971]_.

    Parameters
    ----------
    z : complex np.ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : float
        Half-bandwidth of the DOS of the face-centered cubic lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/8`.

    Returns
    -------
    gf_z : complex np.ndarray or complex
        Value of the face-centered cubic lattice Green's function

    References
    ----------
    .. [morita1971] Morita, T., Horiguchi, T., 1971. Calculation of the Lattice
       Green’s Function for the bcc, fcc, and Rectangular Lattices. Journal of
       Mathematical Physics 12, 986–992. https://doi.org/10.1063/1.1665693

    Examples
    --------
    >>> ww = np.linspace(-1.6, 1.6, num=501, dtype=complex)
    >>> gf_ww = gt.lattice.fcc.gf_z(ww, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.axvline(-0.5, color='black', linewidth=0.8)
    >>> _ = plt.axvline(0, color='black', linewidth=0.8)
    >>> _ = plt.axhline(0, color='black', linewidth=0.8)
    >>> _ = plt.plot(ww.real, gf_ww.real, label=r"$\Re G$")
    >>> _ = plt.plot(ww.real, gf_ww.imag, '--', label=r"$\Im G$")
    >>> _ = plt.ylabel(r"$G*D$")
    >>> _ = plt.xlabel(r"$\omega/D$")
    >>> _ = plt.xlim(left=ww.real.min(), right=ww.real.max())
    >>> _ = plt.legend()
    >>> plt.show()

    """
    D = half_bandwidth / 2
    z = np.asarray(1 / D * z)
    retarded = z.imag > 0
    z = np.where(retarded, np.conj(z), z)  # calculate advanced only, and use symmetry
    zp1 = z + 1
    zp1_pow = _signed_sqrt(zp1)**-3
    sum1 = 4 * _signed_sqrt(z) * zp1_pow
    sum2 = (z - 1) * _signed_sqrt(z - 3) * zp1_pow
    m_p = 0.5*(1 + sum1 - sum2)  # eq. (2.11)
    m_m = 0.5*(1 - sum1 - sum2)  # eq. (2.11)
    kii = np.asarray(_u_ellipk(m_p))
    kii[m_p.imag < 0] += 2j*_u_ellipk(1 - m_p[m_p.imag < 0])  # eq (2.17)
    gf = 4 / (np.pi**2 * D * zp1) * _u_ellipk(m_m) * kii  # eq (2.16)
    return np.where(retarded, np.conj(gf), gf)  # return to retarded by symmetry


def hilbert_transform(xi, half_bandwidth):
    r"""Hilbert transform of non-interacting DOS of the face-centered cubic lattice.

    The Hilbert transform is defined

    .. math:: \tilde{D}(ξ) = ∫_{-∞}^{∞}dϵ \frac{DOS(ϵ)}{ξ − ϵ}

    The lattice Hilbert transform is the same as the non-interacting Green's
    function.

    Parameters
    ----------
    xi : complex np.ndarray or complex
        Point at which the Hilbert transform is evaluated
    half_bandwidth : float
        half-bandwidth of the DOS of the 3D face-centered cubic lattice

    Returns
    -------
    hilbert_transform : complex np.ndarray or complex
        Hilbert transform of `xi`.

    Notes
    -----
    Relation between nearest neighbor hopping `t` and half-bandwidth `D`

    .. math:: 8t = D

    See Also
    --------
    gftool.lattice.fcc.gf_z

    """
    return gf_z(xi, half_bandwidth)


def dos(eps, half_bandwidth):
    r"""DOS of non-interacting 3D face-centered cubic lattice.

    Has a van Hove singularity at `z=-half_bandwidth/2` (divergence) and at
    `z=0` (continuous but not differentiable).

    Parameters
    ----------
    eps : float np.ndarray or float
        DOS is evaluated at points `eps`.
    half_bandwidth : float
        Half-bandwidth of the DOS, DOS(`eps` < -0.5*`half_bandwidth`) = 0,
        DOS(1.5*`half_bandwidth` < `eps`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/8`

    Returns
    -------
    dos : float np.ndarray or float
        The value of the DOS.

    See Also
    --------
    gftool.lattice.fcc.dos_mp : multi-precision version suitable for integration

    References
    ----------
    .. [morita1971] Morita, T., Horiguchi, T., 1971. Calculation of the Lattice
       Green’s Function for the bcc, fcc, and Rectangular Lattices. Journal of
       Mathematical Physics 12, 986–992. https://doi.org/10.1063/1.1665693

    Examples
    --------
    >>> eps = np.linspace(-1.6, 1.6, num=501)
    >>> dos = gt.lattice.fcc.dos(eps, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.axvline(0, color='black', linewidth=0.8)
    >>> _ = plt.axvline(-0.5, color='black', linewidth=0.8)
    >>> _ = plt.plot(eps, dos)
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"DOS * $D$")
    >>> _ = plt.ylim(bottom=0)
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.show()

    """
    eps = np.asarray(eps)
    singular = eps == -0.5*half_bandwidth
    finite = (-0.5*half_bandwidth < eps) & (eps < 1.5*half_bandwidth) & ~singular
    dos_ = np.zeros_like(eps)
    dos_[finite] = 1 / np.pi * gf_z(eps[finite], half_bandwidth=half_bandwidth).imag
    dos_[singular] = np.infty
    return abs(dos_)  # at 0.5D wrong sign


# ∫dϵ ϵ^m DOS(ϵ) for half-bandwidth D=1
# from: integral of dos_mp with mp.workdps(100)
# for m in range(0, 21, 1):
#     with mp.workdps(100):
#         print(mp.quad(lambda eps: eps**m * dos_mp(eps), [mp.mpf('-0.5'), 0, 1])
# rational numbers obtained by mp.identify
dos_moment_coefficients = {
    0: 1,
    1: 0,
    2: 3/16,
    3: 3/32,
    4: 135/1024,
    5: 135/1024,
    6: 0.1611328125,
    7: 0.1922607421875,
    8: 0.24070143699646,
    9: 0.305163860321045,
    10: 0.394462153315544,
    11: 0.516299419105052,
    12: 0.683690124191343,
    13: 0.913928582333027,
    14: 1.23181895411108,
    15: 1.67210463207448,
    16: 2.28395076888283,
    17: 3.13686893977359,
    18: 4.32941325997849,
    19: 6.00152324929046,
    20: 8.35226611969712,
}


def dos_moment(m, half_bandwidth):
    """Calculate the `m` th moment of the face-centered cubic DOS.

    The moments are defined as :math:`∫dϵ ϵ^m DOS(ϵ)`.

    Parameters
    ----------
    m : int
        The order of the moment.
    half_bandwidth : float
        Half-bandwidth of the DOS of the 3D face-centered cubic lattice.

    Returns
    -------
    dos_moment : float
        The `m` th moment of the 3D face-centered cubic DOS.

    Raises
    ------
    NotImplementedError
        Currently only implemented for a few specific moments `m`.

    See Also
    --------
    gftool.lattice.fcc.dos

    """
    try:
        return dos_moment_coefficients[m] * half_bandwidth**m
    except KeyError as keyerr:
        raise NotImplementedError('Calculation of arbitrary moments not implemented.') from keyerr


def _signed_mp_sqrt(eps):
    """Square root with correct sign for fcc lattice."""
    if eps >= 0:
        return mp.sqrt(eps)
    return -1j*mp.sqrt(-eps)


def dos_mp(eps, half_bandwidth=1):
    r"""Multi-precision DOS of non-interacting 3D face-centered cubic lattice.

    Has a van Hove singularity at `z=-half_bandwidth/2` (divergence) and at
    `z=0` (continuous but not differentiable).

    This function is particularity suited to calculate integrals of the form
    :math:`∫dϵ DOS(ϵ)f(ϵ)`. If you have problems with the convergence,
    consider using :math:`∫dϵ DOS(ϵ)[f(ϵ)-f(-1/2)] + f(-1/2)` to avoid the
    singularity.

    Parameters
    ----------
    eps : mpmath.mpf or mpf_like
        DOS is evaluated at points `eps`.
    half_bandwidth : mpmath.mpf or mpf_like
        Half-bandwidth of the DOS, DOS(`eps` < -0.5*`half_bandwidth`) = 0,
        DOS(1.5*`half_bandwidth` < `eps`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/8`

    Returns
    -------
    dos_mp : mpmath.mpf
        The value of the DOS.

    See Also
    --------
    gftool.lattice.fcc.dos : vectorized version suitable for array evaluations

    References
    ----------
    .. [morita1971] Morita, T., Horiguchi, T., 1971. Calculation of the Lattice
       Green’s Function for the bcc, fcc, and Rectangular Lattices. Journal of
       Mathematical Physics 12, 986–992. https://doi.org/10.1063/1.1665693

    Examples
    --------
    Calculate integrals:

    >>> from mpmath import mp
    >>> unit = mp.quad(gt.lattice.fcc.dos_mp, [-0.5, 0, 1.5])
    >>> mp.identify(unit)
    '1'

    >>> eps = np.linspace(-1.6, 1.6, num=501)
    >>> dos_mp = [gt.lattice.fcc.dos_mp(ee, half_bandwidth=1) for ee in eps]
    >>> dos_mp = np.array(dos_mp, dtype=np.float64)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.axvline(0, color='black', linewidth=0.8)
    >>> _ = plt.axvline(-0.5, color='black', linewidth=0.8)
    >>> _ = plt.plot(eps, dos_mp)
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"DOS * $D$")
    >>> _ = plt.ylim(bottom=0)
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.show()

    """
    D = mp.mpf(half_bandwidth) * mp.mpf('1/2')
    eps = mp.mpf(eps) / D
    if 3 < eps < -1:
        return mp.mpf('0')
    if eps == -1:
        return mp.inf
    epsp1 = eps + 1
    epsp1_pow = _signed_mp_sqrt(epsp1)**-3
    sum1 = 4 * _signed_mp_sqrt(eps) * epsp1_pow
    sum2 = (eps - 1) * _signed_mp_sqrt(eps - 3) * epsp1_pow
    m_p = mp.mpf('0.5')*(1 + sum1 - sum2)  # eq. (2.11)
    m_m = mp.mpf('0.5')*(1 - sum1 - sum2)  # eq. (2.11)
    kii = mp.ellipk(m_p)
    if m_p.imag < 0:
        kii += 2j * mp.ellipk(1 - m_p)  # eq (2.17)
    return abs(4 / (mp.pi**3 * D * epsp1) * (_u_ellipk(m_m) * kii).imag)  # eq (2.16)
