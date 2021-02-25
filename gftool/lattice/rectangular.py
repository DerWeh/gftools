"""2D rectangular lattice."""
from functools import partial

import numpy as np

from mpmath import fp

ellipk_z = np.frompyfunc(partial(fp.ellipf, np.pi/2), 1, 1)


def gf_z(z, gamma):
    r"""Local Green's function of the 2D rectangular lattice.

    .. math:: G(z) = \frac{1}{π} ∫_0^π \frac{dϕ}{\sqrt{(t - γ \cos ϕ)^2 - 1}}

    where the integral is the complete elliptic integral of first kind.
    See [morita1971]_.

    Parameters
    ----------
    z : complex np.ndarray or complex
        Green's function is evaluated at complex frequency `z`.

    Returns
    -------
    gf_z : complex ndarray or complex
        Value of the square lattice Green's function

    .. [morita1971] Morita, T., Horiguchi, T., 1971. Calculation of the Lattice
       Green’s Function for the bcc, fcc, and Rectangular Lattices.
       Journal of Mathematical Physics 12, 986–992.
       https://doi.org/10.1063/1.1665693

    Examples
    --------
    >>> ww = np.linspace(-1.5, 1.5, num=500)
    >>> gf_ww = gt.lattice.square.gf_z(ww, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.axhline(0, color='black', linewidth=0.8)
    >>> _ = plt.plot(ww, gf_ww.real, label=r"$\Re G$")
    >>> _ = plt.plot(ww, gf_ww.imag, '--', label=r"$\Im G$")
    >>> _ = plt.ylabel(r"$G*D$")
    >>> _ = plt.xlabel(r"$\omega/D$")
    >>> _ = plt.xlim(left=ww.min(), right=ww.max())
    >>> _ = plt.legend()
    >>> plt.show()

    """
    D = 1 / (1 + gamma)
    z = z / D
    k1 = 4*gamma / (z**2 - (gamma - 1)**2)
    elliptic = ellipk_z(k1)
    try:
        elliptic = elliptic.astype(np.complex)
    except AttributeError:  # elliptic no array, thus no conversion necessary
        pass
    k1sqrt = 2 * gamma**0.5 / np.lib.scimath.sqrt(1 - (gamma - 1)**2/z**2) / z
    gf_z = gamma**-0.5 / np.pi / D * k1sqrt * elliptic
    return gf_z
