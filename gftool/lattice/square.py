r"""2D square lattice.

The dispersion of the 2D square lattice is given by

.. math:: ϵ_{k_x, k_y} = 2t [\cos(k_x) + \cos(k_y)]

which takes values in :math:`ϵ_{k_x, k_y} ∈ [-4t, +4t] = [-D, +D]`.

:half_bandwidth: The half_bandwidth corresponds to a nearest neighbor hopping
                 of `t=D/4`

"""
import numpy as np

from mpmath import mp
from scipy.special import ellipkm1

from gftool._util import _u_ellipk


def gf_z(z, half_bandwidth):
    r"""Local Green's function of the 2D square lattice.

    .. math::
        G(z) = \frac{2}{πz} ∫^{π/2}_{0} \frac{dϕ}{\sqrt{1 - (D/z)^2 \cos^2ϕ}}

    where :math:`D` is the half bandwidth and the integral is the complete
    elliptic integral of first kind. See [economou2006]_.

    Parameters
    ----------
    z : complex np.ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : float
        Half-bandwidth of the DOS of the square lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/4`

    Returns
    -------
    gf_z : complex np.ndarray or complex
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
    >>> _ = plt.axhline(0, color='black', linewidth=0.8)
    >>> _ = plt.plot(ww, gf_ww.real, label=r"$\Re G$")
    >>> _ = plt.plot(ww, gf_ww.imag, '--', label=r"$\Im G$")
    >>> _ = plt.ylabel(r"$G*D$")
    >>> _ = plt.xlabel(r"$\omega/D$")
    >>> _ = plt.xlim(left=ww.min(), right=ww.max())
    >>> _ = plt.legend()
    >>> plt.show()

    """
    z_rel_inv = half_bandwidth/z
    elliptic = _u_ellipk(z_rel_inv**2)
    gf_z = 2./np.pi/half_bandwidth*z_rel_inv*elliptic
    return gf_z


def hilbert_transform(xi, half_bandwidth):
    r"""Hilbert transform of non-interacting DOS of the square lattice.

    The Hilbert transform is defined

    .. math:: \tilde{D}(ξ) = ∫_{-∞}^{∞}dϵ \frac{DOS(ϵ)}{ξ − ϵ}

    The lattice Hilbert transform is the same as the non-interacting Green's
    function.

    Parameters
    ----------
    xi : complex np.ndarray or complex
        Point at which the Hilbert transform is evaluated
    half_bandwidth : float
        half-bandwidth of the DOS of the 2D square lattice

    Returns
    -------
    hilbert_transform : complex np.ndarray or complex
        Hilbert transform of `xi`.

    Notes
    -----
    Relation between nearest neighbor hopping `t` and half-bandwidth `D`

    .. math:: 4t = D

    See Also
    --------
    gftool.lattice.square.gf_z

    """
    return gf_z(xi, half_bandwidth)


def dos(eps, half_bandwidth):
    r"""DOS of non-interacting 2D square lattice.

    Has a van Hove singularity (logarithmic divergence) at `eps = 0`.

    Parameters
    ----------
    eps : float np.ndarray or float
        DOS is evaluated at points `eps`.
    half_bandwidth : float
        Half-bandwidth of the DOS, DOS(| `eps` | > `half_bandwidth`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/4`

    Returns
    -------
    dos : float np.ndarray or float
        The value of the DOS.

    See Also
    --------
    gftool.lattice.square.dos_mp : multi-precision version suitable for integration

    References
    ----------
    .. [economou2006] Economou, E. N. Green's Functions in Quantum Physics.
       Springer, 2006.

    Examples
    --------
    >>> eps = np.linspace(-1.1, 1.1, num=500)
    >>> dos = gt.lattice.square.dos(eps, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.plot(eps, dos)
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"DOS * $D$")
    >>> _ = plt.axvline(0, color='black', linewidth=0.8)
    >>> _ = plt.ylim(bottom=0)
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.show()

    """
    eps_rel = np.asarray(eps / half_bandwidth)
    dos = np.zeros_like(eps_rel)
    nonzero = abs(eps_rel) <= 1
    elliptic = ellipkm1(eps_rel[nonzero]**2)  # on real axis we can use fast scipy Implementation
    dos[nonzero] = 2 / np.pi**2 / half_bandwidth * elliptic
    return dos


# ∫dϵ ϵ^m DOS(ϵ) for half-bandwidth D=1
# from: integral of dos_mp with mp.workdps(100)
# for m in range(0, 22, 2):
#     with mp.workdps(100):
#         print(mp.quad(lambda eps: 2 * eps**m * dos_mp(eps), [0, 1])
# rational numbers obtained by mp.identify
dos_moment_coefficients = {
    2: 0.25,
    4: 9/64,
    6: 25/256,
    8: (35/128)**2,
    10: (63/256)**2,
    12: 0.0508890151977539,
    14: 0.0438787937164307,
    16: 0.0385653460398316,
    18: 0.0343993364367634,
    20: 0.031045401134179,
}


def dos_moment(m, half_bandwidth):
    """Calculate the `m` th moment of the square DOS.

    The moments are defined as :math:`∫dϵ ϵ^m DOS(ϵ)`.

    Parameters
    ----------
    m : int
        The order of the moment.
    half_bandwidth : float
        Half-bandwidth of the DOS of the 2D square lattice.

    Returns
    -------
    dos_moment : float
        The `m` th moment of the 2D square DOS.

    Raises
    ------
    NotImplementedError
        Currently only implemented for a few specific moments `m`.

    See Also
    --------
    gftool.lattice.square.dos

    """
    if m % 2:  # odd moments vanish due to symmetry
        return 0
    try:
        return dos_moment_coefficients[m] * half_bandwidth**m
    except KeyError as keyerr:
        raise NotImplementedError('Calculation of arbitrary moments not implemented.') from keyerr


def dos_mp(eps, half_bandwidth=1):
    r"""Multi-precision DOS of non-interacting 2D square lattice.

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
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/4`

    Returns
    -------
    dos_mp : mpmath.mpf
        The value of the DOS.

    See Also
    --------
    gftool.lattice.square.dos : vectorized version suitable for array evaluations

    References
    ----------
    .. [economou2006] Economou, E. N. Green's Functions in Quantum Physics.
       Springer, 2006.

    Examples
    --------
    Calculate integrals:

    >>> from mpmath import mp
    >>> mp.quad(gt.lattice.square.dos_mp, [-1, 0, 1])
    mpf('1.0')

    >>> eps = np.linspace(-1.1, 1.1, num=500)
    >>> dos_mp = [gt.lattice.square.dos_mp(ee, half_bandwidth=1) for ee in eps]
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
        return mp.mpf('0')
    # around 0 we have to double precision for `1 - eps**2` to resolve around singularity
    with mp.workdps(mp.dps*2, normalize_output=True):
        mm = -mp.powm1(eps / half_bandwidth, mp.mpf('2'))
        return 2 / (mp.pi**2 * half_bandwidth) * mp.ellipk(mm)


def stress_trafo(xi, half_bandwidth):
    r"""Single pole integration over the stress tensor function.

    In analogy to the Hilbert transformation, we define the stress
    tensor transformation as

    .. math:: T(ξ) = ∫dϵ \tilde{Φ}_{xx}(ϵ)/(ξ - ϵ)

    with the stress tensor function

    .. math:: \tilde{Φ}_{xx}(ϵ) ≔ ∑_k 𝜕^2/𝜕k_x^2 δ(ϵ - ϵ_k) = -0.5 * ϵ * DOS(ϵ)

    Parameters
    ----------
    xi : complex or complex array_like
        Point of evaluation of the transformation
    half_bandwidth : float
        Half-bandwidth of the square lattice.

    References
    ----------
    .. [arsenault2013] Arsenault, L.-F., Tremblay, A.-M.S., 2013. Transport
       functions for hypercubic and Bethe lattices. Phys. Rev. B 88, 205109.
       https://doi.org/10.1103/PhysRevB.88.205109

    """
    return -0.5 * (xi*gf_z(xi, half_bandwidth=half_bandwidth) - 1)
