r"""3D face-centered cubic (fcc) lattice.

The dispersion of the 3D face-centered cubic lattice is given by

.. math::

   ϵ_{k_x,k_y,k_z} = 4t [\cos(k_x/2)\cos(k_y/2) + \cos(k_x/2)\cos(k_z/2) + \cos(k_y/2) \cos(k_z/2)]

which takes values in :math:`ϵ_{k_x, k_y, k_z} ∈ [-4t, +12t] = [-0.5D, +1.5D]`.

:half_bandwidth: The half_bandwidth corresponds to a nearest neighbor hopping
                 of `t=D/8`.

"""
import numpy as np

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
    >>> _ = plt.plot(eps, dos)
    >>> _ = plt.axvline(0, color='black', linewidth=0.8)
    >>> _ = plt.axvline(-0.5, color='black', linewidth=0.8)
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
    return dos_
