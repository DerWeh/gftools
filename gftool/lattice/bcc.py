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
