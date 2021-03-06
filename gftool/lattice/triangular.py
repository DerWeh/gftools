r"""2D triangular lattice.

The dispersion of the 2D triangular lattice is given by

.. math:: ϵ_{k_x, k_y} = t [\cos(2k_x) + 2 \cos(k_x)\cos(k_y)]

which takes values :math:`ϵ_{k_x, k_y} ∈ [-1.5t, 3t]`.

:half_bandwidth: The half-bandwidth `D` corresponds to a nearest neighbor hopping
                 of `t=4D/9`.

"""
import numpy as np

from scipy.special import ellipk

from gftool._util import _u_ellipk


def _signed_sqrt(z):
    """Square root with correct sign for triangular lattice."""
    sign = np.where((z.real < 0) & (z.imag < 0), -1, 1)
    return sign * np.lib.scimath.sqrt(z)


def gf_z(z, half_bandwidth):
    r"""Local Green's function of the 2D triangular lattice.

    Note, that the spectrum is asymmetric and in :math:`[-2D/3, 4D/3]`,
    where :math:`D` is the half-bandwidth.
    The Green's function is evaluated as complete elliptic integral of first
    kind, see [horiguchi1972]_.

    Parameters
    ----------
    z : complex np.ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : float
        Half-bandwidth of the DOS of the triangular lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping
        :math:`t=4D/9`.

    Returns
    -------
    gf_z : complex np.ndarray or complex
        Value of the triangular lattice Green's function

    References
    ----------
    .. [horiguchi1972] Horiguchi, T., 1972. Lattice Green’s Functions for the
       Triangular and Honeycomb Lattices. Journal of Mathematical Physics 13,
       1411–1419. https://doi.org/10.1063/1.1666155

    Examples
    --------
    >>> ww = np.linspace(-1.5, 1.5, num=500, dtype=complex) + 1e-64j
    >>> gf_ww = gt.lattice.triangular.gf_z(ww, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.axhline(0, color='black', linewidth=0.8)
    >>> _ = plt.axvline(-2/3, color='black', linewidth=0.8)
    >>> _ = plt.axvline(+4/3, color='black', linewidth=0.8)
    >>> _ = plt.plot(ww.real, gf_ww.real, label=r"$\Re G$")
    >>> _ = plt.plot(ww.real, gf_ww.imag, '--', label=r"$\Im G$")
    >>> _ = plt.ylabel(r"$G*D$")
    >>> _ = plt.xlabel(r"$\omega/D$")
    >>> _ = plt.xlim(left=ww.real.min(), right=ww.real.max())
    >>> _ = plt.legend()
    >>> plt.show()

    """
    D = half_bandwidth * 4 / 9
    z = 1.0 / D * np.asarray(z)
    shape = z.shape
    z = z.reshape(-1)
    advanced = z.imag < 0
    z = np.where(advanced, np.conj(z), z)  # calculate retarded only, and use symmetry
    singular = D * z == -1  # try singularity which needs to be avoided
    z[singular] = 0  # mock value to avoid errors
    rr = _signed_sqrt(2*z + 3)
    gg = 4.0 / (_signed_sqrt(rr - 1)**3 * _signed_sqrt(rr + 3))  # eq (2.9)
    kk = _signed_sqrt(rr) * gg  # eq (2.11)
    mm = kk**2
    K = np.asarray(_u_ellipk(mm))
    # eqs (2.22) and eq (2.18), fix correct plane
    K[kk.imag > 0] += 2j*_u_ellipk(1 - mm[kk.imag > 0])
    gf_z = 1 / np.pi / D * gg * K  # eq (2.6)
    gf_z[singular] = 0 - 1j*np.infty
    return np.where(advanced, np.conj(gf_z), gf_z).reshape(shape)  # return to advanced by symmetry


def dos(eps, half_bandwidth):
    r"""DOS of non-interacting 2D triangular lattice.

    The DOS diverges at `-4/9*half_bandwidth`.
    The DOS is evaluated as complete elliptic integral of first kind,
    see [kogan2021]_.

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

    References
    ----------
    .. [kogan2021] Kogan, E. and Gumbs, G. (2021) Green’s Functions and DOS for
       Some 2D Lattices. Graphene, 10, 1-12.
       https://doi.org/10.4236/graphene.2021.101001.

    Examples
    --------
    >>> eps = np.linspace(-1.5, 1.5, num=500)
    >>> dos = gt.lattice.triangular.dos(eps, half_bandwidth=1)

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
    D = half_bandwidth * 4 / 9
    eps = 1.0 / D * eps
    dos = np.zeros_like(eps)
    region1 = (-1.5 <= eps) & (eps <= -1)
    rr = np.sqrt(2*eps[region1] + 3)
    z0 = (rr + 1)**3 * (3 - rr) / 4
    z1 = 4 * rr
    dos[region1] = 1 / np.sqrt(z0) * ellipk(z1/z0)
    region2 = (-1 <= eps) & (eps <= +3)
    rr = np.sqrt(2*eps[region2] + 3)
    z0 = 4 * rr
    z1 = (rr + 1)**3 * (3 - rr) / 4
    dos[region2] = 1 / np.sqrt(z0) * ellipk(z1/z0)
    return 2 / np.pi**2 / D * dos
