"""1D lattice.

:half_bandwidth: The half_bandwidth corresponds to a nearest neighbor hopping
                 of `t=D/2`

"""
import numpy as np


def gf_z(z, half_bandwidth):
    r"""Local Green's function of the 1D lattice.

    .. math::
       G(z) = \frac{1}{2 π} ∫_{-π}^{π}\frac{dϕ}{z - D\cos(ϕ)}

    where :math:`D` is the half bandwidth. The integral can be evaluated in the
    complex plane along the unit circle. See [economou2006]_.

    Parameters
    ----------
    z : complex ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : float
        Half-bandwidth of the DOS of the 1D lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`

    References
    ----------
    .. [economou2006] Economou, E. N. Green's Functions in Quantum Physics.
       Springer, 2006.

    Examples
    --------
    >>> ww = np.linspace(-1.5, 1.5, num=500)
    >>> gf_ww = gt.lattice.onedim.gf_z(ww, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.plot(ww, gf_ww.real, label=r"$\Re G$")
    >>> _ = plt.plot(ww, gf_ww.imag, '--', label=r"$\Im G$")
    >>> _ = plt.xlabel(r"$\omega/D$")
    >>> _ = plt.axhline(0, color='black', linewidth=0.8)
    >>> _ = plt.axvline(0, color='black', linewidth=0.8)
    >>> _ = plt.xlim(left=ww.min(), right=ww.max())
    >>> _ = plt.legend()
    >>> plt.grid()
    >>> plt.show()

    """
    z_rel_inv = half_bandwidth / z
    return 1. / half_bandwidth * z_rel_inv / np.lib.scimath.sqrt(1 - z_rel_inv**2)


def hilbert_transform(xi, half_bandwidth):
    r"""Hilbert transform of non-interacting DOS of the 1d lattice.

    The Hilbert transform is defined

    .. math::
        \tilde{D}(ξ) = ∫_{-∞}^{∞}dϵ \frac{DOS(ϵ)}{ξ − ϵ}

    The lattice Hilbert transform is the same as the non-interacting Green's
    function.

    Parameters
    ----------
    xi : complex ndarray or complex
        Point at which the Hilbert transform is evaluated
    half_bandwidth : float
        half-bandwidth of the DOS of the 1D lattice

    Returns
    -------
    hilbert_transform : complex ndarray or complex
        Hilbert transform of `xi`.

    Notes
    -----
    Relation between nearest neighbor hopping `t` and half-bandwidth `D`

    .. math:: 2t = D

    See Also
    --------
    gftool.lattice.onedim.gf_z

    """
    return gf_z(xi, half_bandwidth)


def dos(eps, half_bandwidth):
    r"""DOS of non-interacting 1D lattice.

    Parameters
    ----------
    eps : float ndarray or float
        DOS is evaluated at points `eps`.
    half_bandwidth : float
        Half-bandwidth of the DOS, DOS(| `eps` | > `half_bandwidth`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`

    Returns
    -------
    dos : float ndarray or float
        The value of the DOS.

    Examples
    --------
    >>> eps = np.linspace(-1.1, 1.1, num=500)
    >>> dos = gt.lattice.onedim.dos(eps, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.plot(eps, dos)
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"DOS * $D$")
    >>> _ = plt.axhline(0, color='black', linewidth=0.8)
    >>> _ = plt.axvline(0, color='black', linewidth=0.8)
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.grid()
    >>> plt.show()

    """
    eps_rel = np.asarray(eps / half_bandwidth)
    dos = np.zeros_like(eps_rel)
    nonzero = (abs(eps_rel) <= 1) | np.iscomplex(eps)
    dos[nonzero] = 1. / (np.pi*half_bandwidth) / np.sqrt(1 - eps_rel[nonzero]**2)
    return dos


# ∫dϵ ϵ^m DOS(ϵ) for half-bandwidth D=1
# from: wolframalpha
dos_moment_coefficients = {
    0: 1.,
    2: 0.5,
    4: 0.375,
    6: 5. / 16.,
    8: 35. / 128.,
    10: 63. / 256.,
}


def dos_moment(m, half_bandwidth):
    """Calculate the `m` th moment of the 1D DOS.

    The moments are defined as :math:`∫dϵ ϵ^m DOS(ϵ)`.

    Parameters
    ----------
    m : int
        The order of the moment.
    half_bandwidth : float
        Half-bandwidth of the DOS of the 1D lattice.

    Returns
    -------
    dos_moment : float
        The `m` th moment of the 1D DOS.

    Raises
    ------
    NotImplementedError
        Currently only implemented for a few specific moments `m`.

    See Also
    --------
    gftool.lattice.onedim.dos

    """
    if m % 2:  # odd moments vanish due to symmetry
        return 0
    try:
        return dos_moment_coefficients[m] * half_bandwidth**m
    except KeyError as keyerr:
        raise NotImplementedError('Calculation of arbitrary moments not implemented.') from keyerr
