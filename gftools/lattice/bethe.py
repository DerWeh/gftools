"""Bethe lattice with infinite coordination number.

This is in fact no real lattice, but a tree. It corresponds to a semi-circular
DOS.

"""
import numpy as np

_PRECISE_TYPES = {np.dtype(np.complex256), np.dtype(np.float128)}


def gf_z(z, half_bandwidth):
    """Local Green's function of Bethe lattice for infinite coordination number.

    Parameters
    ----------
    z : complex ndarray or complex
        Green's function is evaluated at complex frequency `z`
    half_bandwidth : float
        half-bandwidth of the DOS of the Bethe lattice
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`

    Returns
    -------
    gf_z : complex ndarray or complex
        Value of the Green's function

    TODO: source

    """
    z_rel = np.array(z / half_bandwidth, dtype=np.complex256)
    try:
        complex_pres = np.complex256 if z.dtype in _PRECISE_TYPES else np.complex
    except AttributeError:
        complex_pres = np.complex
    gf_z = 2./half_bandwidth*z_rel*(1 - np.sqrt(1 - z_rel**-2))
    return gf_z.astype(dtype=complex_pres, copy=False)


def gf_d1_z(z, half_bandwidth):
    """First derivative of local Green's function of Bethe lattice for infinite coordination number.

    Parameters
    ----------
    z : complex ndarray or complex
        Green's function is evaluated at complex frequency `z`
    half_bandwidth : float
        half-bandwidth of the DOS of the Bethe lattice
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`

    Returns
    -------
    gf_d1_z : complex ndarray or complex
        Value of the derivative of the Green's function

    See Also
    --------
    gf_z

    """
    z_rel = z / half_bandwidth
    sqrt = (1 - z_rel**-2)
    # return 2. * (1. - sqrt - (z_rel**-2 / sqrt))
    return 2. * (1 - 1/sqrt)


def hilbert_transform(xi, half_bandwidth):
    r"""Hilbert transform of non-interacting DOS of the Bethe lattice.

    FIXME: the lattice Hilbert transform is the same as the non-interacting
        Green's function.

    The Hilbert transform

    .. math::
        \tilde{D}(ξ) = ∫_{-∞}^{∞}dϵ \frac{DOS(ϵ)}{ξ − ϵ}

    takes for Bethe lattice in the limit of infinite coordination number the
    explicit form

    .. math::
        \tilde{D}(ξ) = 2*(ξ - s\sqrt{ξ^2 - D^2})/D^2

    with :math:`s=sgn[ℑ{ξ}]`.
    See `Georges et al`_.


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

    Note
    ----
    Relation between nearest neighbor hopping `t` and half-bandwidth `D`

    .. math::
        2t = D

    References
    ----------
    .. [1] Georges et al: https://doi.org/10.1103/RevModPhys.68.13

    """
    return gf_z(xi, half_bandwidth)


def dos(eps, half_bandwidth):
    """DOS of non-interacting Bethe lattice for infinite coordination number.

    Parameters
    ----------
    eps : float ndarray or float
        DOS is evaluated at points `eps`.
    half_bandwidth : float
        Half-bandwidth of the DOS, DOS(| `eps` | > `half_bandwidth`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`

    Returns
    -------
    result : float ndarray or float
        The value of the DOS.

    """
    D2 = half_bandwidth * half_bandwidth
    eps2 = eps*eps
    mask = eps2 < D2
    try:
        result = np.empty_like(eps)
        result[~mask] = 0
    except IndexError:  # eps is scalar
        if mask:
            return np.sqrt(D2 - eps2) / (0.5 * np.pi * D2)
        return 0.  # outside of bandwidth
    else:
        result[mask] = np.sqrt(D2 - eps2[mask]) / (0.5 * np.pi * D2)
        return result


# ∫dϵ ϵ^m DOS(ϵ) for half-bandwidth D=1
# from:
# https://www.wolframalpha.com/input/?i=integrate+sqrt(1+-+x%5E2)%2F(.5*pi)+*x%5E%7B2*n%7Ddx+from+-1+to+1+assuming+n+is+integer
dos_moment_coefficients = {
    2: 0.25,
    4: 0.125,
    6: 0.078125,
    8: 0.0546875,
    # not sure if results form wolframalpha are accurate enought, calculated them with mpmath
    # 10: 0.0410156,
    # 12: 0.0322266,
    # 14: 0.0261841,
    # 16: 0.0218201,
    # 18: 0.0185471,
    # 20: 0.0160179,
}


def dos_moment(m, half_bandwidth):
    """Calculate the `m`th moment of the Bethe DOS.

    The moments are defined as `math`:∫dϵ ϵ^m DOS(ϵ):.

    Parameters
    ----------
    m : int
        The order of the moment.
    half_bandwidth : float
        Half-bandwidth of the DOS of the Bethe lattice.

    Returns
    -------
    dos_moment : float
        The `m`th moment of the Bethe DOS.

    Raises
    ------
    NotImplementedError
        Currently only implemented for a few specific moments `m`.

    """
    if m % 2:  # odd moments vanish due to symmetry
        return 0
    try:
        return dos_moment_coefficients[m] * half_bandwidth**m
    except KeyError:
        raise NotImplementedError('Calculation of arbitrary moments not implemented.')
