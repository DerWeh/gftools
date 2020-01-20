"""2D square lattice.

:half_bandwidth: The half_bandwidth corresponds to a nearest neighbor hopping
                 of `t=D/4`

"""
from functools import partial

import numpy as np
from scipy.special import ellipkm1
from mpmath import fp

ellipk_z = partial(fp.ellipf, np.pi/2)


def gf_z(z, half_bandwidth):
    r"""Local Green's function of the 2D square lattice.

    .. math::
        G(z) = \frac{2}{πz} ∫^{π/2}_{0} \frac{dϕ}{\sqrt{1 - (D/z)^2 \cos^2ϕ}}

    where :math:`D` is the half bandwidth and the integral is the complete
    elliptic integral of first kind. See [economou2006]_.

    Parameters
    ----------
    z : complex ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : float
        Half-bandwidth of the DOS of the square lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/4`

    Returns
    -------
    gf_z : complex ndarray or complex
        Value of the square lattice Green's function

    References
    ----------
    .. [economou2006] Economou, E. N. Green's Functions in Quantum Physics.
       Springer, 2006.

    Examples
    --------
    >>> ww = np.linspace(-1.5, 1.5, num=500)
    >>> gf_ww = gt.lattice.square.gf_z(ww, half_bandwidth=1)

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
    z_rel_inv = half_bandwidth/z
    elliptic = np.frompyfunc(ellipk_z, 1, 1)(z_rel_inv**2)
    try:
        elliptic = elliptic.astype(np.complex)
    except AttributeError:  # elliptic no array, thus no conversion necessary
        pass
    gf_z = 2./np.pi/half_bandwidth*z_rel_inv*elliptic
    return gf_z


def hilbert_transform(xi, half_bandwidth):
    r"""Hilbert transform of non-interacting DOS of the square lattice.

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
        half-bandwidth of the DOS of the Bethe lattice

    Returns
    -------
    hilbert_transfrom : complex ndarray or complex
        Hilbert transform of `xi`.

    Notes
    -----
    Relation between nearest neighbor hopping `t` and half-bandwidth `D`

    .. math::
        4t = D

    See Also
    --------
    gf_z

    """
    return gf_z(xi, half_bandwidth)


def dos(eps, half_bandwidth):
    r"""DOS of non-interacting 2D square lattice.

    Parameters
    ----------
    eps : float ndarray or float
        DOS is evaluated at points `eps`.
    half_bandwidth : float
        Half-bandwidth of the DOS, DOS(| `eps` | > `half_bandwidth`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/4`

    Returns
    -------
    dos : float ndarray or float
        The value of the DOS.

    Examples
    --------
    >>> eps = np.linspace(-1.1, 1.1, num=500)
    >>> dos = gt.lattice.square.dos(eps, half_bandwidth=1)

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
    nonzero = abs(eps_rel) <= 1
    elliptic = ellipkm1(eps_rel[nonzero]**2)  # on the real axis we can use fast scipy Implementation
    dos[nonzero] = 2 / np.pi**2 / half_bandwidth * elliptic
    return dos


# ∫dϵ ϵ^m DOS(ϵ) for half-bandwidth D=1
# from: wolframalpha, to integral in python to assert accuracy
dos_moment_coefficients = {
    2: 0.25,
    4: 0.140625,
    6: 0.0976563,
}


def dos_moment(m, half_bandwidth):
    """Calculate the `m` th moment of the square DOS.

    The moments are defined as :math:`∫dϵ ϵ^m DOS(ϵ)`.

    Parameters
    ----------
    m : int
        The order of the moment.
    half_bandwidth : float
        Half-bandwidth of the DOS of the Bethe lattice.

    Returns
    -------
    dos_moment : float
        The `m` th moment of the Bethe DOS.

    Raises
    ------
    NotImplementedError
        Currently only implemented for a few specific moments `m`.

    See Also
    --------
    dos

    """
    if m % 2:  # odd moments vanish due to symmetry
        return 0
    try:
        return dos_moment_coefficients[m] * half_bandwidth**m
    except KeyError:
        raise NotImplementedError('Calculation of arbitrary moments not implemented.')
