"""3D simple cubic lattice.

:half_bandwidth: The half_bandwidth corresponds to a nearest neighbor hopping
                 of `t=D/6`

"""
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np

from mpmath import mp
from scipy import integrate, interpolate


class _DOSContainer:
    """Manage pre-computed DOS."""

    file_name = "scubic_dos.npz"
    van_hove = 1.0/3.0

    def __init__(self):
        self._data = None
        self._interp_1 = None
        self._interp_2 = None

    def __call__(self, eps, half_bandwidth):
        r"""DOS of non-interacting 3D simple cubic lattice.

        Has a van Hove singularity (continuous but not differentiable) at
        `abs(eps) = D/3`.

        Parameters
        ----------
        eps : float ndarray or float
            DOS is evaluated at points `eps`.

        Returns
        -------
        dos : float ndarray or float
            The value of the DOS.

        See Also
        --------
        gftool.lattice.scubic.dos_mp : Calculation of DOS using multi-precision integration

        Notes
        -----
        This function is a spline for precomputed vales calculated by `dos_mp`.
        The used data-points are available at `dos_container.data`

        Examples
        --------
        >>> eps = np.linspace(-1.1, 1.1, num=500)
        >>> dos = gt.lattice.scubic.dos(eps, half_bandwidth=1)

        >>> import matplotlib.pyplot as plt
        >>> _ = plt.plot(eps, dos)
        >>> _ = plt.xlabel(r"$\epsilon/D$")
        >>> _ = plt.ylabel(r"DOS * $D$")
        >>> _ = plt.axvline(+1/3, color="black", linestyle="--")
        >>> _ = plt.axvline(-1/3, color="black", linestyle="--")
        >>> _ = plt.axvline(0, color='black', linewidth=0.8)
        >>> _ = plt.ylim(bottom=0)
        >>> _ = plt.xlim(left=eps.min(), right=eps.max())
        >>> plt.show()

        """
        eps_rel = np.asanyarray(eps / half_bandwidth)
        eps_rel = abs(eps_rel)
        dos_ = np.zeros_like(eps_rel)
        domain1 = eps_rel <= self.van_hove
        dos_[domain1] = self.interp_1(eps_rel[domain1])  # pylint: disable=not-callable
        domain2 = (self.van_hove < eps_rel) & (eps_rel < 1)
        dos_[domain2] = self.interp_2(eps_rel[domain2])  # pylint: disable=not-callable
        return dos_ / half_bandwidth

    def fdos(self, eps: float, half_bandwidth):
        """Faster evaluation for `float` `eps`."""
        # pylint: disable=protected-access
        eps_rel = abs(eps / half_bandwidth)
        if eps_rel <= self.van_hove:
            return self.interp_1._spline(eps_rel).item() / half_bandwidth
        if self.van_hove < eps_rel < 1:
            return self.interp_2._spline(eps_rel).item() / half_bandwidth
        return 0

    @property
    def data(self) -> dict:
        """Return data of the pre-computed dos.

        It contains the following keys:

        'x1', 'dos1'
            Energy points and DOS at this energy points on the interval
            `[0, self.van_hove]`.

        'x2', 'dos2'
            Energy points and DOS at this energy points on the interval
            `[self.van_hove, 1]`.

        """
        if self._data is None:
            with np.load(Path(__file__).parent / self.file_name) as data:
                self._data = dict(data)
        return self._data

    @property
    def interp_1(self) -> Callable[[float], float]:
        """Return cubic interpolation on the interval `[0, self.van_hove]`."""
        if self._interp_1 is None:
            self._interp_1 = interpolate.interp1d(
                self.data['x1'], y=self.data['dos1'], kind='cubic',
                assume_sorted=True,
                # fill_value="extrapolate",
            )
        return self._interp_1

    @property
    def interp_2(self) -> Callable[[float], float]:
        """Return cubic interpolation on the interval `[self.van_hove, 1]`."""
        if self._interp_2 is None:
            self._interp_2 = interpolate.interp1d(
                self.data['x2'], y=self.data['dos2'], kind='cubic',
                assume_sorted=True,
                # fill_value="extrapolate",
            )
        return self._interp_2


dos_container = _DOSContainer()

dos = dos_container.__call__


def dos_mp(eps, half_bandwidth=1, maxdegree: int = None):
    r"""Multi-precision DOS of non-interacting 3D simple cubic lattice.

    Has a van Hove singularity (continuous but not differentiable) at
    `abs(eps) = D/3`.

    For accurate results the working precision should be increased, to avoid
    errors in the integration weights, e.g.:

    >>> from mpmath import mp
    >>> with mp.workdps(30):
    ...     dos, err = gt.lattice.scubic.dos_mp(mp.mpf('0.2'))

    Parameters
    ----------
    eps : mp.mpf or mpf_like
        Energy point at which the DOS is evaluated.
        Will be converted to a multi-precision float `mp.mpf`.
    half_bandwidth : mp.mpf or mpf_like
        Half-bandwidth of the DOS.

    Returns
    -------
    dos, dos_err : mp.mpf
        Value of the DOS and estimate for the integration error.

    Other Parameters
    ----------------
    maxdegree : int, optional
        Maximum degree of the quadrature rule to try before quitting.
        Passed to `mpmath.mp.quad`.

    See Also
    --------
    gfool.lattice.scubic.dos : Spline for precomputed DOS for fast evaluation.


    Examples
    --------
    >>> from mpmath import mp
    >>> eps = np.linspace(-1.1, 1.1, num=101)
    >>> with mp.workdps(30):
    ...     dos = [gt.lattice.scubic.dos_mp(ee)[0] for ee in eps]

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.plot(eps, dos, 'x-')
    >>> _ = plt.xlabel(r"$\epsilon/D$")
    >>> _ = plt.ylabel(r"DOS * $D$")
    >>> _ = plt.axvline(1/3, color="black", linestyle="--")
    >>> _ = plt.axvline(-1/3, color="black", linestyle="--")
    >>> _ = plt.axvline(0, color='black', linewidth=0.8)
    >>> _ = plt.ylim(bottom=0)
    >>> _ = plt.xlim(left=eps.min(), right=eps.max())
    >>> plt.show()

    References
    ----------
    .. [economou2006] Economou, E. N. Green's Functions in Quantum Physics.
       Springer, 2006.

    """
    eps = abs(mp.mpf(eps)) / half_bandwidth
    quad = partial(mp.quad, error=True, maxdegree=maxdegree)
    pre_factor = 3 / mp.pi**2 / half_bandwidth

    def _integrand(phi):
        eps_ = 3*eps - mp.cospi(phi)
        if eps_ > 2:
            return 0
        return mp.ellipk(1 - mp.mpf(0.25)*eps_**2)

    if 3*eps < 1:
        # somewhere I read it would be good to shift the singularity to 0,
        # else there are round-off errors
        # Don't know how to do it here, so I cheat by excluding the singularity
        singularity = mp.acos(3 * eps) / mp.pi
        delta = mp.eps**(1/mp.mpf(2))
        integ1 = quad(_integrand, [0, singularity - delta])
        integ2 = quad(_integrand, [singularity + delta, 1])
        return [pre_factor * (i1 + i2) for i1, i2 in zip(integ1, integ2)]

    if eps <= 1:
        upper = mp.acos(3*eps - 2)/mp.pi
        integ = quad(_integrand, [0, upper])
        return [pre_factor * ii for ii in integ]

    if eps > 1:
        return mp.mpf('0'), mp.mpf('0')


def gf_z(z, half_bandwidth):
    r"""Local Green's function of the 3D cubic lattice.

    It is calculate as the lattice Hilbert transform

    .. :math: ∫dϵ_{-D}^{D} DOS(ϵ)/(z - ϵ)

    where :math:`D` is the half-bandwidth and :math:`DOS` the density of states.
    Note that for `z.imag=0`, the integrand contains a singularity which is
    not explicitly treated here.

    Parameters
    ----------
    z : complex ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : float
        Half-bandwidth of the DOS of the cubic lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/6`

    Returns
    -------
    gf_z : complex ndarray or complex
        Value of the cubic lattice Green's function
    error : float
        Estimate for the integration error.

    Examples
    --------
    >>> ww = np.linspace(-1.5, 1.5, num=500) + 1e-3j
    >>> gf_ww, err = gt.lattice.scubic.gf_z(ww, half_bandwidth=1)

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
    z_rel = z / half_bandwidth
    z_rel2 = z_rel**2

    # we separate the diverging part ∫ dϵ DOS(z.real)/(z - ϵ) to speed up for small z.imag
    dos_realz = dos(z_rel.real, half_bandwidth=1)
    correction = dos_realz * np.log((z_rel + 1.0) / (z_rel - 1.0))

    def integrand(eps):
        return (dos_container.fdos(eps, half_bandwidth=1) - dos_realz) * z_rel / (z_rel2 - eps**2)

    int1, err1 = integrate.quad_vec(integrand, a=0, b=dos_container.van_hove)
    int2, err2 = integrate.quad_vec(integrand, a=dos_container.van_hove, b=1)
    return 2*(int1 + int2)/half_bandwidth + correction, 2*(err1 + err2)/half_bandwidth
