r"""3D body-centered cubic (bcc) lattice.

The dispersion of the 3D simple cubic lattice is given by

.. math:: ϵ_{k_x, k_y, k_z} = t \cos(k_x) \cos(k_y) \cos(k_z)

which takes values in :math:`ϵ_{k_x, k_y, k_z} ∈ [-t, +t] = [-D, +D]`.

:half_bandwidth: The half_bandwidth corresponds to a nearest neighbor hopping
                 of `t=D/2`

"""
import numpy as np

from numpy.lib.scimath import sqrt

from gftool._util import _u_ellipk


def gf_z(z, half_bandwidth):
    r"""Local Green's function of 3D body-centered cubic (bcc) lattice.

    Has a van Hove singularity at `z=0` (divergence).

    Implements equations (2.1) and (2.4) from [morita1971]_

    Parameters
    ----------
    z : complex np.ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : float
        Half-bandwidth of the DOS of the body-centered cubic lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`

    Returns
    -------
    gf_z : complex np.ndarray or complex
        Value of the body-centered cubic Green's function at complex energy `z`.

    References
    ----------
    .. [morita1971] Morita, T., Horiguchi, T., 1971. Calculation of the Lattice
       Green’s Function for the bcc, fcc, and Rectangular Lattices. Journal of
       Mathematical Physics 12, 986–992. https://doi.org/10.1063/1.1665693

    Examples
    --------
    >>> ww = np.linspace(-1.5, 1.5, num=500)
    >>> gf_ww = gt.lattice.bcc.gf_z(ww, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.axhline(0, color='black', linewidth=0.8)
    >>> _ = plt.plot(ww, gf_ww.real, label=r"$\Re G$")
    >>> _ = plt.plot(ww, gf_ww.imag, '--', label=r"$\Im G$")
    >>> _ = plt.xlabel(r"$\omega/D$")
    >>> _ = plt.ylabel(r"$G*D$")
    >>> _ = plt.xlim(left=ww.min(), right=ww.max())
    >>> _ = plt.legend()
    >>> plt.show()

    """
    z_rel = z / half_bandwidth
    # k = (sqrt(z_rel + 1) - sqrt(z_rel - 1)) / (2*sqrt(z_rel))
    m = 0.5 * (1 - sqrt(1 - z_rel**-2))
    return 4 / (np.pi**2 * z) * _u_ellipk(m)**2


def dos(eps, half_bandwidth):
    r"""DOS of non-interacting 3D body-centered cubic lattice.

    Has a van Hove singularity (logarithmic divergence) at `eps = 0`.

    Parameters
    ----------
    eps : float np.ndarray or float
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
    gftool.lattice.bcc.dos_mp : multi-precision version suitable for integration

    References
    ----------
    .. [morita1971] Morita, T., Horiguchi, T., 1971. Calculation of the Lattice
       Green’s Function for the bcc, fcc, and Rectangular Lattices. Journal of
       Mathematical Physics 12, 986–992. https://doi.org/10.1063/1.1665693

    Examples
    --------
    >>> eps = np.linspace(-1.1, 1.1, num=500)
    >>> dos = gt.lattice.bcc.dos(eps, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.plot(eps, dos)
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"DOS * $D$")
    >>> _ = plt.axvline(0, color='black', linewidth=0.8)
    >>> _ = plt.ylim(bottom=0)
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.show()

    """
    eps = np.asarray(eps)
    eps_rel = eps / half_bandwidth
    dos_ = np.zeros_like(eps)
    singular = eps_rel == 0
    finite = (abs(eps_rel) <= 1) & ~singular
    m = 0.5 - 0.5j*np.sqrt(eps_rel[finite]**-2 - 1)
    Ksqr = _u_ellipk(m)**2
    dos_[finite] = -4 / (np.pi**3 * abs(eps[finite])) * Ksqr.imag
    dos_[singular] = np.infty
    return dos_
