r"""3D body-centered cubic (bcc) lattice.

The dispersion of the 3D body-centered cubic lattice is given by

.. math:: ϵ_{k_x, k_y, k_z} = 8t \cos(k_x) \cos(k_y) \cos(k_z)

which takes values in :math:`ϵ_{k_x, k_y, k_z} ∈ [-8t, +8t] = [-D, +D]`.

:half_bandwidth: The half_bandwidth corresponds to a nearest neighbor hopping
                 of `t=D/8`

"""
import numpy as np

from numpy.lib.scimath import sqrt
from mpmath import mp

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
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/8`

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


def hilbert_transform(xi, half_bandwidth):
    r"""Hilbert transform of non-interacting DOS of the body-centered cubic lattice.

    The Hilbert transform is defined

    .. math:: \tilde{D}(ξ) = ∫_{-∞}^{∞}dϵ \frac{DOS(ϵ)}{ξ − ϵ}

    The lattice Hilbert transform is the same as the non-interacting Green's
    function.

    Parameters
    ----------
    xi : complex np.ndarray or complex
        Point at which the Hilbert transform is evaluated
    half_bandwidth : float
        half-bandwidth of the DOS of the 3D body-centered cubic lattice

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
    gftool.lattice.bcc.gf_z

    """
    return gf_z(xi, half_bandwidth)


def dos(eps, half_bandwidth):
    r"""DOS of non-interacting 3D body-centered cubic lattice.

    Has a van Hove singularity (logarithmic divergence) at `eps = 0`.

    Parameters
    ----------
    eps : float np.ndarray or float
        DOS is evaluated at points `eps`.
    half_bandwidth : float
        Half-bandwidth of the DOS, DOS(| `eps` | > `half_bandwidth`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/8`

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
    # identity `K(m) = (1/m) [K(m) + iK'(1/m)]` could be used to avoid 1/0
    m = 0.5 - 0.5j*np.sqrt(eps_rel[finite]**-2 - 1)
    Ksqr = _u_ellipk(m)**2
    dos_[finite] = -4 / (np.pi**3 * abs(eps[finite])) * Ksqr.imag
    dos_[singular] = np.infty
    return dos_


# ∫dϵ ϵ^m DOS(ϵ) for half-bandwidth D=1
# from: integral of dos_mp with mp.workdps(100)
# for m in range(0, 22, 2):
#     with mp.workdps(100):
#         print(mp.quad(lambda eps: 2 * eps**m * dos_mp(eps), [0, 1])
# rational numbers obtained by mp.identify
dos_moment_coefficients = {
    2: 1/8,
    4: 27/512,
    6: 5**3 / 2**12,
    8: 0.020444393157959,
    10: 0.0149039626121521,
    12: 0.0114798462018371,
    14: 0.00919140747282654,
    16: 0.0075734863820287,
    18: 0.00638006491682219,
    20: 0.00547010815806043,
}


def dos_moment(m, half_bandwidth):
    """Calculate the `m` th moment of the body-centered cubic DOS.

    The moments are defined as :math:`∫dϵ ϵ^m DOS(ϵ)`.

    Parameters
    ----------
    m : int
        The order of the moment.
    half_bandwidth : float
        Half-bandwidth of the DOS of the 3D body-centered cubic lattice.

    Returns
    -------
    dos_moment : float
        The `m` th moment of the 3D body-centered cubic DOS.

    Raises
    ------
    NotImplementedError
        Currently only implemented for a few specific moments `m`.

    See Also
    --------
    gftool.lattice.bcc.dos

    """
    if m % 2:  # odd moments vanish due to symmetry
        return 0
    try:
        return dos_moment_coefficients[m] * half_bandwidth**m
    except KeyError as keyerr:
        raise NotImplementedError('Calculation of arbitrary moments not implemented.') from keyerr


def dos_mp(eps, half_bandwidth=1):
    r"""Multi-precision DOS of non-interacting 3D body-centered lattice.

    Has a van Hove singularity (logarithmic divergence) at `eps = 0`.

    This function is particularity suited to calculate integrals of the form
    :math:`∫dϵ DOS(ϵ)f(ϵ)`. If you have problems with the convergence,
    consider using :math:`∫dϵ DOS(ϵ)[f(ϵ)-f(0)] + f(0)` to avoid the singularity.

    Parameters
    ----------
    eps : mpmath.mpf or mpf_like
        DOS is evaluated at points `eps`.
    half_bandwidth : mpmath.mpf or mpf_like
        Half-bandwidth of the DOS, DOS(| `eps` | > `half_bandwidth`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/8`

    Returns
    -------
    dos_mp : mpmath.mpf
        The value of the DOS.

    See Also
    --------
    gftool.lattice.bcc.dos : vectorized version suitable for array evaluations

    References
    ----------
    .. [morita1971] Morita, T., Horiguchi, T., 1971. Calculation of the Lattice
       Green’s Function for the bcc, fcc, and Rectangular Lattices. Journal of
       Mathematical Physics 12, 986–992. https://doi.org/10.1063/1.1665693

    Examples
    --------
    Calculate integrals:

    >>> from mpmath import mp
    >>> mp.quad(gt.lattice.bcc.dos_mp, [-1, 0, 1])
    mpf('1.0')

    >>> eps = np.linspace(-1.1, 1.1, num=500)
    >>> dos_mp = [gt.lattice.bcc.dos_mp(ee, half_bandwidth=1) for ee in eps]
    >>> dos_mp = np.array(dos_mp, dtype=np.float64)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.plot(eps, dos_mp)
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"DOS * $D$")
    >>> _ = plt.axvline(0, color='black', linewidth=0.8)
    >>> _ = plt.ylim(bottom=0)
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.show()

    """
    eps, half_bandwidth = mp.mpf(eps), mp.mpf(half_bandwidth)
    if mp.fabs(eps) > half_bandwidth:
        return mp.mpf("0")
    if eps == mp.mpf("0"):
        return mp.inf
    eps_rel = eps / half_bandwidth
    m = mp.mpf("0.5") - mp.mpc("0", "0.5")*mp.sqrt(mp.powm1(eps_rel, -2))
    Ksqr = mp.ellipk(m)**2
    return -4 / (mp.pi**3 * mp.fabs(eps)) * Ksqr.imag
