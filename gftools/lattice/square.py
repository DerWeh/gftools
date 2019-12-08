"""2D square lattice.

:half_bandwidth: The half_bandwidth corresponds to a nearest neighbor hopping
                 of `t=D/4`

"""
from functools import partial

import numpy as np
from mpmath import fp

ellipk = partial(fp.ellipf, np.pi/2)


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
    >>> plt.plot(ww, gf_ww.real, label=r"$\Re G$")
    >>> plt.plot(ww, gf_ww.imag, '--', label=r"$\Im G$")
    >>> plt.xlabel(r"$\omega/D$")
    >>> plt.axhline(0, color='black', linewidth=0.8)
    >>> plt.axvline(0, color='black', linewidth=0.8)
    >>> plt.xlim(left=ww.min(), right=ww.max())
    >>> plt.grid()
    >>> plt.legend()
    >>> plt.show()

    """
    z_rel_inv = half_bandwidth/z
    elliptic = np.frompyfunc(ellipk, 1, 1)(z_rel_inv**2)
    try:
        elliptic = elliptic.astype(np.complex)
    except AttributeError:  # elliptic no array, thus no conversion necessary
        pass
    gf_z = 2./np.pi/z*elliptic
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
    >>> plt.plot(eps, dos)
    >>> plt.xlabel(r"$\epsilon/D$")
    >>> plt.ylabel(r"DOS * $D$")
    >>> plt.axhline(0, color='black', linewidth=0.8)
    >>> plt.axvline(0, color='black', linewidth=0.8)
    >>> plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.grid()
    >>> plt.legend()
    >>> plt.show()

    """
    eps_ = np.asarray(eps).reshape(-1)
    dos = np.zeros_like(eps_)
    neg = (eps_ > -half_bandwidth) & (eps_ <= 0.)
    dos[+neg] = +gf_z(eps_[+neg], half_bandwidth).imag
    dos[~neg] = -gf_z(eps_[~neg], half_bandwidth).imag
    return 1./np.pi*dos.reshape(eps.shape)


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
