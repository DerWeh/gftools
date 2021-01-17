"""3D simple cubic lattice.

:half_bandwidth: The half_bandwidth corresponds to a nearest neighbor hopping
                 of `t=D/6`

"""
import logging
import warnings

from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np

from mpmath import mp
from scipy import integrate, interpolate
from scipy.integrate._quad_vec import _max_norm

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


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

    def dos_d1(self, eps, half_bandwidth):
        """1st derivative of DOS of non-interacting 3D simple cubic lattice."""
        eps_rel = np.asanyarray(eps / half_bandwidth)
        eps_rel = abs(eps_rel)
        dos_ = np.zeros_like(eps_rel)
        domain1 = eps_rel <= self.van_hove
        # pylint: disable=not-callable,protected-access
        dos_[domain1] = self.interp_1._spline.derivative()(eps_rel[domain1])[..., 0]
        domain2 = (self.van_hove < eps_rel) & (eps_rel < 1)
        dos_[domain2] = self.interp_2._spline.derivative()(eps_rel[domain2])[..., 0]
        return np.sign(eps)*dos_ / half_bandwidth**2

    def fdos(self, eps: float, half_bandwidth):
        """Faster evaluation for `float` `eps`."""
        # pylint: disable=protected-access
        eps_rel = abs(eps / half_bandwidth)
        output = np.empty((1, 1), dtype=np.float_)
        if eps_rel <= self.van_hove:
            self.interp_1._spline._evaluate(np.array([eps_rel]), 0, False, out=output)
            return output.item() / half_bandwidth
        if self.van_hove < eps_rel < 1:
            self.interp_2._spline._evaluate(np.array([eps_rel]), 0, False, out=output)
            return output.item() / half_bandwidth
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


def gf_z(z, half_bandwidth, error="warn", **quad_kwds):
    r"""Local Green's function of the 3D cubic lattice.

    It is calculate as the lattice Hilbert transform

    .. :math: ∫dϵ_{-D}^{D} DOS(ϵ)/(z - ϵ)

    where :math:`D` is the half-bandwidth and :math:`DOS` the density of states.
    Note that for `z.imag=0`, the integrand contains a singularity which is
    not explicitly treated here.
    The required time depends strongly on the `z.imag`. Around `1e-3 < z.imag < 1e-5`,
    the function becomes slow (slowest for 1e-5). For larger and smaller values,
    the function is reasonable fast.

    Parameters
    ----------
    z : complex ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : float
        Half-bandwidth of the DOS of the cubic lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/6`
    error : {"warn", "raise", "ignore", "return"}
        How to preceded with integration error estimate (default: "warn").
        If `err="return"` the error will be returned with the Green's function.
    quad_kwds
        Keyword arguments passed to `scipy.integrate.quad_vec`

    Returns
    -------
    gf_z : complex ndarray or complex
        Value of the cubic lattice Green's function
    err : float
        Estimate for the integration error. Only given if `error="return"`.

    Examples
    --------
    >>> ww = np.linspace(-1.5, 1.5, num=500) + 1e-3j
    >>> gf_ww = gt.lattice.scubic.gf_z(ww, half_bandwidth=1)

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
    error = error.lower()
    if error not in {"warn", "raise", "ignore", "return"}:
        raise ValueError(f"Unknown argument 'error={error}'")

    gf, err = _gf_z(z/half_bandwidth, **quad_kwds)
    gf /= half_bandwidth
    err /= half_bandwidth

    # handle integration error
    # BEGIN>: copied from scipy.integrate.quad_vec
    norm = quad_kwds.get("norm", "2")
    norm_funcs = {
        None: lambda: _max_norm,
        "max": _max_norm,
        "2": np.linalg.norm
    }
    if callable(norm):
        norm_func = norm
    else:
        norm_func = norm_funcs[norm]
    # <END
    epsrel = quad_kwds.get("epsrel", 1e-8)
    epsabs = quad_kwds.get("epsabs", 1e-16)
    converged = err < max(epsabs, epsrel*norm_func(gf))
    (LOGGER.debug if converged else LOGGER.warn)("Integration error of Green's function: %s", err)

    if error == "return":
        return gf, err
    if error == "warn" and not converged:
        warnings.warn(f"Integration not sufficiently converged: err={err}.",
                      category=RuntimeWarning)
    if error == "raise" and not converged:
        raise RuntimeError(f"Integration not sufficiently converged: err={err}.")
    return gf


def _gf_z(z, **quad_kwds):
    """Perform actual calculation for `gf_z`."""
    z2 = z**2

    # ∫ dϵ DOS(z.real)/(z - ϵ)
    # for small imaginary part, 1/(z - ϵ) becomes strongly peak
    # to speed up integrals for small z.imag, we expand around this ϵ=z.real
    dos_realz = dos(z.real, half_bandwidth=1)
    log = np.log((z + 1.0) / (z - 1.0))
    correction0 = dos_realz * log
    factor = 0.5  # heuristically have of the first order correction is best
    dos_d1_realz = factor*dos_container.dos_d1(z.real, half_bandwidth=1)
    correction1 = dos_d1_realz * ((z - np.conj(z))*log - 2)
    zabs2 = z * np.conj(z)

    def integrand(eps):
        numer = (dos_container.fdos(eps, half_bandwidth=1) - dos_realz) * z
        numer -= dos_d1_realz * (eps**2 - zabs2)
        return numer / (z2 - eps**2)

    int1, err1 = integrate.quad_vec(integrand, a=0, b=dos_container.van_hove, **quad_kwds)
    int2, err2 = integrate.quad_vec(integrand, a=dos_container.van_hove, b=1, **quad_kwds)
    return 2*(int1 + int2) + correction0 + correction1, 2*(err1 + err2)


# ∫dϵ ϵ^m DOS(ϵ) for half-bandwidth D=1
# from: mp quad integration
# with mp.workdps(50):
#     res = 2*mp.quad(lambda eps: eps**2*scubic.dos_mp(eps)[0], [0, mp.mpf(1/3), 1])
dos_moment_coefficients = {
    2: 1/6,  # identified by mp.identify
    4: 5/72,  # identified by mp.identify
    6: 0.03986625514403292181,
    8: 0.02663108710562414266,
    10: 0.01939193244170096022,
    12: 0.01492852797570661654,
    14: 0.01194895308081000525,
    16: 0.009843770445314749191,
    18: 0.008291560006168067136,
    20: 0.007108354149086696661,
}


def dos_moment(m, half_bandwidth):
    """Calculate the `m` th moment of the simple cubic DOS.

    The moments are defined as :math:`∫dϵ ϵ^m DOS(ϵ)`.

    Parameters
    ----------
    m : int
        The order of the moment.
    half_bandwidth : float
        Half-bandwidth of the DOS of the simple cubic lattice.

    Returns
    -------
    dos_moment : float
        The `m` th moment of the simple cubic DOS.

    Raises
    ------
    NotImplementedError
        Currently only implemented for a few specific moments `m`.

    See Also
    --------
    gftool.lattice.scubic.dos

    """
    if m % 2:  # odd moments vanish due to symmetry
        return 0
    try:
        return dos_moment_coefficients[m] * half_bandwidth**m
    except KeyError as keyerr:
        raise NotImplementedError('Calculation of arbitrary moments not implemented.') from keyerr
