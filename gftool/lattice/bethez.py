"""Bethe lattice for general coordination number Z.

In the limit of infinite coordination number `Z=∞`, this becomes `gftool.lattice.bethe`,
in the opposite limit of minimal coordination number `Z=2``, this is `gftool.lattice.onedim`.

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
    >>> _ = plt.ylabel(r"$G*D$")
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


def dos(eps, half_bandwidth, coordination):
    r"""DOS of non-interacting Bethe lattice for `coordination`.

    Parameters
    ----------
    eps : float ndarray or float
        DOS is evaluated at points `eps`.
    half_bandwidth : float
        Half-bandwidth of the DOS, DOS(| `eps` | > `half_bandwidth`) = 0.
    coordination : int
        Coordination number of the Bethe lattice.

    Returns
    -------
    dos : float ndarray or float
        The value of the DOS.

    See also
    --------
    gftool.lattice.bethe.dos : case for `coordination=np.infty`
    gftool.lattice.onedim.dos : case for `coordination=2`

    Examples
    --------
    >>> eps = np.linspace(-1.1, 1.1, num=500)
    >>> dos = gt.lattice.bethez.dos(eps, half_bandwidth=1, coordination=9)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.plot(eps, dos)
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"DOS * $D$")
    >>> _ = plt.axvline(0, color='black', linewidth=0.8)
    >>> _ = plt.ylim(bottom=0)
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.show()

    """
    assert coordination > 1
    eps_rel = np.asarray(eps / half_bandwidth)
    Z = coordination
    dos_ = np.zeros_like(eps_rel)
    nonzero = (abs(eps_rel) <= 1) | np.iscomplex(eps)
    eps2 = eps_rel[nonzero]**2
    factor = 2. * Z * (Z - 1) / (half_bandwidth * np.pi)
    dos_[nonzero] = factor * np.sqrt(1 - eps2) / (Z**2 - 4*(Z - 1)*eps2)
    return dos_
