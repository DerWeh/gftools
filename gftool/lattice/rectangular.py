r"""2D rectangular lattice.

The dispersion of the 2D rectangular lattice is given by

.. math:: ϵ_{k_x, k_y} = 2t [\cos(k_x) + γ\cos(k_y)]

where :math:`γ` is the `scale`.
The DOS has a singularity at :math:`2t(γ-1)=D(γ-1)/(γ+1)`.

:half_bandwidth: The half-bandwidth `D` corresponds to a nearest neighbor hopping
                 of `t=D/2/(scale + 1)`
:scale: Relative scale of the different hopping `t_1 = scale*t_2`.

"""
import numpy as np

from numpy.lib import scimath

from gftool._util import _u_ellipk


def gf_z(z, half_bandwidth, scale):
    r"""Local Green's function of the 2D rectangular lattice.

    .. math:: G(z) = \frac{1}{π} ∫_0^π \frac{dϕ}{\sqrt{(z - γ \cos ϕ)^2 - 1}}

    where :math:`γ` is the `scale`, the hopping is chosen `t=1`, the
    `half_bandwidth` is :math:`D=2(1+γ)`.
    The integral is the complete elliptic integral of first kind.
    See [morita1971]_.

    Parameters
    ----------
    z : complex np.ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : float
        Half-bandwidth of the DOS of the rectangular lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping
        :math:`t=D/2/(scale+1)`.
    scale : float
        Relative scale of the different hoppings :math:`t_1=scale*t_2`.
        `scale=1` corresponds to the square lattice.

    Returns
    -------
    gf_z : complex np.ndarray or complex
        Value of the rectangular lattice Green's function

    See Also
    --------
    gftool.lattice.square.gf_z : Green's function in the limit `scale=1`

    References
    ----------
    .. [morita1971] Morita, T., Horiguchi, T., 1971. Calculation of the Lattice
       Green’s Function for the bcc, fcc, and Rectangular Lattices.
       Journal of Mathematical Physics 12, 986–992.
       https://doi.org/10.1063/1.1665693

    Examples
    --------
    >>> ww = np.linspace(-1.5, 1.5, num=500, dtype=complex)
    >>> gf_ww = gt.lattice.rectangular.gf_z(ww, half_bandwidth=1, scale=2)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.axhline(0, color='black', linewidth=0.8)
    >>> _ = plt.plot(ww.real, gf_ww.real, label=r"$\Re G$")
    >>> _ = plt.plot(ww.real, gf_ww.imag, '--', label=r"$\Im G$")
    >>> _ = plt.ylabel(r"$G*D$")
    >>> _ = plt.xlabel(r"$\omega/D$")
    >>> _ = plt.xlim(left=ww.real.min(), right=ww.real.max())
    >>> _ = plt.legend()
    >>> plt.show()

    """
    D = half_bandwidth / (1 + scale)
    z = z / D
    sm1p2 = (scale - 1)**2
    k1 = 4*scale / (z**2 - sm1p2)
    elliptic = _u_ellipk(k1)
    z_inv = 1 / z
    k1sqrt = 1 / scimath.sqrt(1 - sm1p2*z_inv**2)
    gf_z = 2 / np.pi / D * z_inv * k1sqrt * elliptic
    return gf_z


def hilbert_transform(xi, half_bandwidth, scale):
    r"""Hilbert transform of non-interacting DOS of the rectangular lattice.

    The Hilbert transform is defined

    .. math:: \tilde{D}(ξ) = ∫_{-∞}^{∞}dϵ \frac{DOS(ϵ)}{ξ − ϵ}

    The lattice Hilbert transform is the same as the non-interacting Green's
    function.

    Parameters
    ----------
    xi : complex np.ndarray or complex
        Point at which the Hilbert transform is evaluated
    half_bandwidth : float
        Half-bandwidth of the DOS of the 2D rectangular lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping
        :math:`t=D/2/(scale+1)`.
    scale : float
        Relative scale of the different hoppings :math:`t_1=scale*t_2`.
        `scale=1` corresponds to the square lattice.

    Returns
    -------
    hilbert_transform : complex np.ndarray or complex
        Hilbert transform at `xi`.

    Notes
    -----
    Relation between nearest neighbor hopping `t`, scale `γ` and half-bandwidth `D`

    .. math:: 2(γ+1)t = D

    See Also
    --------
    gftool.lattice.rectangular.gf_z

    """
    return gf_z(xi, half_bandwidth, scale=scale)


def dos(eps, half_bandwidth, scale):
    r"""DOS of non-interacting 2D rectangular lattice.

    Parameters
    ----------
    eps : float np.ndarray or float
        DOS is evaluated at points `eps`.
    half_bandwidth : float
        Half-bandwidth of the DOS, DOS(| `eps` | > `half_bandwidth`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping
        :math:`t=D/2/(scale+1)`.
    scale : float
        Relative scale of the different hoppings :math:`t_1=scale*t_2`.
        `scale=1` corresponds to the square lattice.

    Returns
    -------
    dos : float np.ndarray or float
        The value of the DOS.

    Examples
    --------
    >>> eps = np.linspace(-1.1, 1.1, num=500)
    >>> dos = gt.lattice.rectangular.dos(eps, half_bandwidth=1, scale=2)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.plot(eps, dos)
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"DOS * $D$")
    >>> _ = plt.axvline(0, color='black', linewidth=0.8)
    >>> _ = plt.ylim(bottom=0)
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.show()

    """
    # same code as gf_z, using the itentity for the elliptic integral
    # `K(m) = K(m/(m-1)).conj() / (1-m)**0.5` real `m`
    # we don't have to worry about the correct signs here, we just take the positive one
    eps = np.asarray(eps / half_bandwidth)
    dos_ = np.zeros_like(eps)
    # eps = 1 divergeces in the current formulation and has to be callculated as limit
    eps[abs(eps) == 1] = 1 - np.finfo(float).eps  # avoid 1, inaccurate fix
    nonzero = abs(eps) <= 1
    eps = eps[nonzero]  # calculate only relevant region
    kmod = 4*scale / (scale + 1)**2 / (1 - eps**2)
    elliptic = _u_ellipk(kmod)
    factor = 1.0 / np.pi**2 * (1 + scale) / scale**0.5 / half_bandwidth
    dos_[nonzero] = factor * np.sqrt(kmod) * np.conj(elliptic).real
    return abs(dos_)
