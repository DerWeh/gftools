"""Bathe lattice for general coordination number Z.

In the limit of infinite coordination number `Z=∞`, this becomes `gftool.lattice.bethe`,
in the opposite limit of minimal coordination number `Z=2``, this is `gftool.lattice.onedim`.

Note, that unlike the other `gftool.lattice`, this module is only a very limited
implementation, currently only `gf_z` is available.

"""
import numpy as np


def gf_z(z, half_bandwidth, coordination):
    r"""Local Green's function of Bethe lattice for `coordination`.

    .. math:: G(z) = 2 (Z - 1) / z / ((Z - 2) + Z\sqrt{1 - D^2/z^2})

    where :math:`D` is the `half_bandwidth` and :math:`Z` the`coordination`.
    See [economou2006]_.

    Parameters
    ----------
    z : complex ndarray or complex
        Green's function is evaluated at complex frequency `z`
    half_bandwidth : float
        Half-bandwidth of the DOS of the Bethe lattice.
    coordination : int
        Coordination number of the Bethe lattice.

    Returns
    -------
    gf_z : complex ndarray or complex
        Value of the Bethe Green's function

    See also
    --------
    gftool.lattice.bethe.gf_z : case for `coordination=np.infty`
    gftool.lattice.onedim.gf_z : case for `coordination=2`

    References
    ----------
    .. [economou2006] Economou, E. N. Green's Functions in Quantum Physics.
       Springer, 2006.

    Examples
    --------
    >>> ww = np.linspace(-1.5, 1.5, num=500)
    >>> gf_ww = gt.lattice.bethez.gf_z(ww, half_bandwidth=1, coordination=9)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.plot(ww, gf_ww.real, label=r"$\Re G$")
    >>> _ = plt.plot(ww, gf_ww.imag, '--', label=r"$\Im G$")
    >>> _ = plt.xlabel(r"$\omega/D$")
    >>> _ = plt.xlabel(r"$G*D$")
    >>> _ = plt.axhline(0, color='black', linewidth=0.8)
    >>> _ = plt.xlim(left=ww.min(), right=ww.max())
    >>> _ = plt.legend()
    >>> plt.show()

    """
    assert coordination > 1
    z_rel_inv = half_bandwidth / z
    Z = coordination
    sqrt = np.lib.scimath.sqrt(1 - z_rel_inv**2)
    return 2 * (Z - 1) / half_bandwidth * z_rel_inv / ((Z - 2) + Z*sqrt)
