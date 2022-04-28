r"""Padé based on robust pole finding.

Instead of fitting a rational polynomial, poles and zeros or poles and
corresponding residues are fitted.

The algorithm is based on [ito2018]_ and adjusted to Green's functions and self-
energies. A very short summary can of the algorithm can be found in the appendix
of [weh2020]_.
We assume that we know exactly the high-frequency behavior of the
function we want to continue. Here, we will call it `degree` and we define it
as the behavior of the function `f(z)` for large `abs(z)`:

.. math:: f(z) ≈ z^{degree}.

For the diagonal element's of the one-particle Green's function this is `degree=-1`
for the self-energy it is `degree=0`.

References
----------
.. [ito2018] Ito, S., Nakatsukasa, Y., 2018. Stable polefinding and rational
   least-squares fitting via eigenvalues. Numer. Math. 139, 633–682.
   https://doi.org/10.1007/s00211-018-0948-4
.. [weh2020] Weh, A. et al. Spectral properties of heterostructures containing
   half-metallic ferromagnets in the presence of local many-body correlations.
   Phys. Rev. Research 2, 043263 (2020).
   https://doi.org/10.1103/PhysRevResearch.2.043263

Examples
--------
The function `continuation` provides a high-level interface which can be used
for convenience. Let's consider an optimal example: We know the Green's
function for complex frequencies on the unit (half-)circle. We consider the
Bethe Green's function.

.. plot::
   :format: doctest
   :context: close-figs

   >>> z = np.exp(1j*np.linspace(np.pi, 0, num=252)[1:-1])
   >>> gf_z = gt.bethe_gf_z(z, half_bandwidth=1)
   >>> pade = gt.polepade.continuation(z, gf_z, degree=-1, moments=[1])
   >>> print(f"[{pade.zeros.size}/{pade.poles.size}]")
   [14/15]

We obtain a ``[14/15](z)`` Padé approximant. Let's compare it on the real axis:

.. plot::
   :format: doctest
   :context: close-figs

   >>> import matplotlib.pyplot as plt
   >>> ww = np.linspace(-1.1, 1.1, num=500) + 1e-6j
   >>> gf_ww = gt.bethe_gf_z(ww, half_bandwidth=1)
   >>> pade_ww = pade.eval_polefct(ww)
   >>> __ = plt.axhline(0, color='dimgray', linewidth=0.8)
   >>> __ = plt.axvline(0, color='dimgray', linewidth=0.8)
   >>> __ = plt.plot(ww.real, -gf_ww.imag/np.pi)
   >>> __ = plt.plot(ww.real, -pade_ww.imag/np.pi, '--')
   >>> __ = plt.xlabel(r"$\omega$")
   >>> plt.show()

Beside the band-edge, we get a nice fit. We can also investigate the pole
structure of the fit:

.. plot::
   :format: doctest
   :context: close-figs

   >>> pade.plot()
   >>> plt.show()

Using a grid on the imaginary axis, the fit is of course worse.
Note, that its typically better to continue the self-energy instead of the
Green's function, see appendix of [weh2020]_.
For more control, instead of using `continuation` the elementary functions can
be used:

* `number_poles` to determine the degree of the Padé approximant
* `poles`, `zeros` to calculate the poles and zeros of the approximant
* `residues_ols` to calculate the residues

"""
import logging

from dataclasses import dataclass

import numpy as np

from scipy import linalg
from numpy.polynomial import polynomial

from gftool.basis import PoleFct, ZeroPole
from gftool.linalg import lstsq_ec, orth_compl

LOGGER = logging.getLogger(__name__)


@dataclass
class PadeApprox:
    """Representation of the Padé approximation based on poles.

    Basically the approximation is obtained as `~gftool.basis.PoleFct` as well
    as `~gftool.basis.ZeroPole`. Note however, that those to approximations
    will in general not agree. Nevertheless, for a good approximation they
    should be very similar.

    Parameters
    ----------
    zeros : (..., Nz) complex np.ndarray
        Zeros of the represented function.
    poles, residues : (..., Np) complex np.ndarray
        Poles and the corresponding residues of the represented function.
    amplitude : (...) complex np.ndarray or complex
        The amplitude of the function. This is also the large `abs(z)` limit
        of the function `ZeroPole.eval(z) = amplitude * z**(Nz-Np)`.

    """

    zeros: np.ndarray
    poles: np.ndarray
    residues: np.ndarray
    amplitude: np.ndarray

    def eval_polefct(self, z):
        """Evaluate the `PoleFct` representation."""
        degree = self.zeros.shape[-1] - self.poles.shape[-1]
        assert degree <= 0
        const = self.amplitude if degree == 0 else 0
        return const + PoleFct.eval_z(self, z)

    def eval_zeropole(self, z):
        """Evaluate the `ZeroPole` representation."""
        return ZeroPole.eval(self, z)

    def moments(self, order):
        """Calculate high-frequency moments of `PoleFct` representation."""
        return PoleFct.moments(self, order)

    def plot(self, residue=False, axis=None):
        """Represent the function as scatter plot."""
        import matplotlib as mpl  # pylint: disable=[import-outside-toplevel,import-error]
        import matplotlib.pyplot as plt  # pylint: disable=[import-outside-toplevel,import-error]
        if axis is None:
            axis = plt.gca()
        axis.axhline(0, color='.3')
        axis.axvline(0, color='.3')
        cmap = mpl.cm.get_cmap('cividis_r')
        norm = mpl.colors.Normalize(vmin=min(abs(self.residues)),
                                    vmax=max(abs(self.residues)))
        axis.scatter(self.poles.real, self.poles.imag, label='poles',
                     color=cmap(norm(abs(self.residues))), marker='o')
        if residue:  # indicate residues as error-bars
            axis.quiver(self.poles.real, self.poles.imag, self.residues.real, self.residues.imag,
                        ((abs(self.residues))), norm=norm, cmap=cmap, width=0.004,
                        units='xy', angles='xy', scale_units='xy', scale=1)
        axis.scatter(self.zeros.real, self.zeros.imag, label='zeros', marker='x', color='black')
        axis.set_xlabel(r"$\Re z$")
        axis.set_ylabel(r"$\Im z$")
        axis.legend()
        cbar = plt.colorbar(mappable=mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis)
        cbar.ax.set_ylabel('|residue|')


# TODO: implement linear search, we have a finite N_z
def number_poles(z, fct_z, *, degree=-1, weight=None, n_poles0: int = None,
                 vandermond=polynomial.polyvander) -> int:
    """Estimate the optimal number of poles for a rational approximation.

    The number of poles is determined, such that up to numerical accuracy the
    solution is unique, corresponding to a null-dimension equal to 1 [ito2018]_.

    Parameters
    ----------
    z, fct_z : (N_z) complex np.ndarray
        Variable where function is evaluated and function values.
    degree : int, optional
        The difference of denominator and numerator degree. (default: -1)
        This determines how `fct_z` decays for large `abs(z)`:
        `fct_z → z**degree`. For Green's functions it typically is `-1`, for
        self-energies it typically is `0`.
    weight : (N_z) float np.ndarray, optional
        Weighting of the data points, for a known error `σ` this should be
        `weight = 1./σ`.
    n_poles0 : int, optional
        Starting guess for the number of poles. Can be given to speed up
        calculation if a good estimate is available.
    vandermond : Callable, optional
        Function giving the Vandermond matrix of the chosen polynomial basis.
        Defaults to simple polynomials.

    Returns
    -------
    number_poles : int
        Best guess for optimal number of poles.

    References
    ----------
    .. [ito2018] Ito, S., Nakatsukasa, Y., 2018. Stable polefinding and rational
       least-squares fitting via eigenvalues. Numer. Math. 139, 633–682.
       https://doi.org/10.1007/s00211-018-0948-4

    """
    # pylint: disable=too-many-locals
    tol = np.finfo(fct_z.dtype).eps
    max_n_poles = abs_max_n_poles = (z.size - degree)//2
    if n_poles0 is None:
        n_poles = min(max_n_poles, 50)
    else:
        if n_poles0 > max_n_poles:
            raise ValueError(
                "'n_poles0' to large, system is under determined!"
                "\n#poles + #zeroes = 2*#poles + degree must be smaller than"
                f"number of given points {z.size}"
            )
        n_poles = n_poles0
    assert 2*n_poles + degree < z.size
    while True:
        n_zeros = n_poles + degree
        vander = vandermond(z, deg=max(n_poles, n_zeros)+1)
        numer = vander[..., :n_zeros+1]
        denom = vander[..., :n_poles+1]
        scaling = 1./np.linalg.norm(np.concatenate((denom, numer), axis=-1), axis=-1, keepdims=True)
        if weight is not None:
            scaling *= weight[..., np.newaxis]
        q_fct_x_denom, __ = np.linalg.qr(scaling*fct_z[:, np.newaxis]*denom)
        q_numer, __ = np.linalg.qr(scaling*numer)
        mat = np.concatenate([q_fct_x_denom, q_numer], axis=-1)
        singular_values = np.linalg.svd(mat, compute_uv=False)
        null_dim = np.count_nonzero(singular_values < tol*singular_values[0]*max(mat.shape))
        if null_dim == 1:  # correct number of poles
            return n_poles
        if null_dim == 0:  # too few poles
            if n_poles == abs_max_n_poles:
                raise RuntimeError(
                    f"No solution with {abs_max_n_poles} poles or less could be found!"
                )
            if n_poles == max_n_poles:
                print("Warning: residue is bigger then tolerance: "
                      f"{singular_values[-1]/singular_values[0]}.")
                return n_poles
            # increase number of poles
            n_poles = min(2*n_poles, max_n_poles)
        else:  # already too many poles
            max_n_poles = n_poles - 1
            n_poles = n_poles - (null_dim - degree)//2


def poles(z, fct_z, *, n: int = None, m: int, vandermond=polynomial.polyvander, weight=None):
    """Calculate position of the `m` poles.

    Parameters
    ----------
    z, fct_z : (N_z) complex np.ndarray
        Variable where function is evaluated and function values.
    n, m : int
        Number of zeros and poles of the function.
        For large `z` the function is proportional to `z**(n - m)`.
        (`n` defaults to `m-1`)
    vandermond : Callable, optional
        Function giving the Vandermond matrix of the chosen polynomial basis.
    weight : (N_z) float np.ndarray, optional
        Weighting of the data points, for a known error `σ` this should be
        `weight = 1./σ`.

    Returns
    -------
    poles : (m) complex np.ndarray
        The position of the poles.

    Notes
    -----
    The calculation closely follows [ito2018]_, we just adjust the scaling.

    References
    ----------
    .. [ito2018] Ito, S., Nakatsukasa, Y., 2018. Stable polefinding and rational
       least-squares fitting via eigenvalues. Numer. Math. 139, 633–682.
       https://doi.org/10.1007/s00211-018-0948-4

    """
    # pylint: disable=too-many-locals
    if n is None:
        n = m - 1
    fct_z = fct_z/np.median(fct_z)
    vander = vandermond(z, deg=max(n+1, m))
    numer = -vander[..., :n+1]
    denom = vander[..., :m]
    scaling = 1./np.linalg.norm(np.concatenate((denom, numer), axis=-1), axis=-1, keepdims=True)

    if weight is not None:
        scaling *= weight[..., np.newaxis]
    perp_numer = orth_compl(scaling*numer)
    q_fct_denom, __ = np.linalg.qr(scaling*fct_z[..., np.newaxis]*denom, mode='reduced')
    # q_fct_denom, *__ = spla.qr(D*B1, mode='economic', pivoting=True)
    qtilde_z_fct_denom = perp_numer @ (z[..., np.newaxis] * q_fct_denom)
    qtilde_fct_denom = perp_numer @ q_fct_denom
    __, __, vh = np.linalg.svd(np.concatenate((qtilde_z_fct_denom, qtilde_fct_denom), axis=-1))
    return linalg.eig(vh[:m, :m], vh[:m, m:], right=False)


def zeros(z, fct_z, poles, *, n: int = None, vandermond=polynomial.polyvander, weight=None):
    """Calculate position of `n` zeros given the `poles`.

    Parameters
    ----------
    z, fct_z : (N_z) complex np.ndarray
        Variable where function is evaluated and function values.
    poles : (m) complex np.ndarray
        Position of the poles of the function
    n : int
        Number of zeros.
        For large `z` the function is proportional to `z**(n - m)`.
        (`n` defaults to `m-1`)
    vandermond : Callable, optional
        Function giving the Vandermond matrix of the chosen polynomial basis.
    weight : (N_z) float np.ndarray, optional
        Weighting of the data points, for a known error `σ` this should be
        `weight = 1./σ`.

    Returns
    -------
    zeros : (n) complex np.ndarray
        The position of the zeros.

    Notes
    -----
    The calculation closely follows [ito2018]_, we just adjust the scaling.

    References
    ----------
    .. [ito2018] Ito, S., Nakatsukasa, Y., 2018. Stable polefinding and rational
       least-squares fitting via eigenvalues. Numer. Math. 139, 633–682.
       https://doi.org/10.1007/s00211-018-0948-4

    """
    if n is None:
        n = poles.size - 1
    fct_z = fct_z/np.median(fct_z)
    denom = np.prod(np.subtract.outer(z, poles), axis=-1)
    numer = vandermond(z, deg=n-1)
    scaling = 1./np.linalg.norm(numer, axis=-1, keepdims=True)
    if weight is not None:
        scaling *= weight[..., np.newaxis]
    perp_fct_z_denom = orth_compl(scaling*(fct_z*denom)[:, np.newaxis])
    q_numer, __ = np.linalg.qr(scaling*numer, mode='reduced')
    qtilde_z_numer = perp_fct_z_denom @ (z[..., np.newaxis] * q_numer)
    qtilde_numer = perp_fct_z_denom @ q_numer
    __, __, vh = np.linalg.svd(np.concatenate((qtilde_z_numer, qtilde_numer), axis=-1))
    return linalg.eig(vh[:n, :n], vh[:n, n:], right=False)


def asymptotic(z, fct_z, zeros, poles, weight=None):
    """Calculate large `z` asymptotic from `roots` and `poles`.

    We assume `f(z) = a * np.prod(z - zeros) / np.prod(z - poles)`, therefore
    The asymptotic for large `abs(z)` is `f(z) ≈ a * z**(zeros.size - poles.size)`.

    Parameters
    ----------
    z, fct_z : (N_z) complex np.ndarray
        Variable where function is evaluated and function values.
    zeros, poles : (n), (m) complex np.ndarray
        Position of the zeros and poles of the function.
    weight : (N_z) float np.ndarray, optional
        Weighting of the data points, for a known error `σ` this should be
        `weight = 1./σ`.

    Returns
    -------
    asym, std : float
        Large `z` asymptotic and its standard deviation.

    """
    ratios = fct_z * ZeroPole(zeros, poles).reciprocal(z)
    if weight is None:
        asym = np.mean(ratios, axis=-1)
        std = np.std(ratios, ddof=1, axis=-1)
    else:
        asym = np.average(ratios, weights=weight, axis=-1)
        std = np.average(abs(ratios - asym)**2, weights=weight, axis=-1)
    return asym, std


def residues_ols(z, fct_z, poles, weight=None, moments=()):
    """Calculate the residues using ordinary least square.

    Parameters
    ----------
    z, fct_z : (N_z) complex np.ndarray
        Variable where function is evaluated and function values.
    poles : (M) complex np.ndarray
        Position of the poles of the function
    weight : (N_z) float np.ndarray, optional
        Weighting of the data points, for a known error `σ` this should be
        `weight = 1./σ`.
    moments : (N) float array_like
        Moments of the high-frequency expansion, where
        `f(z) = moments / z**np.arange(1, N+1)` for large `z`.

    Returns
    -------
    residues : (M) complex np.ndarray
        The residues corresponding to the `poles`.
    residual : (1)
        Norm of the residual.

    """
    polematrix = 1./np.subtract.outer(z, poles)
    if weight is not None:
        polematrix *= weight[..., np.newaxis]
        fct_z = fct_z*weight
    moments = np.asarray(moments)
    if moments.shape[-1] == 0:
        return np.linalg.lstsq(polematrix, fct_z, rcond=None)[:2]
    if moments.shape[-1] > poles.size:
        raise ValueError("Too many poles given, system is over constrained. "
                         f"poles: {poles.size}, moments: {moments.shape[-1]}")
    constrain_mat = polynomial.polyvander(poles, deg=moments.shape[-1]-1).T
    resid = lstsq_ec(polematrix, fct_z, constrain_mat, moments)
    return resid, [np.linalg.norm(np.sum(polematrix*resid, axis=-1) - fct_z, ord=2)]


def continuation(z, fct_z, degree=-1, weight=None, moments=(),
                 vandermond=polynomial.polyvander, rotate=None, real_asymp=True) -> PadeApprox:
    """Perform the Padé analytic continuation of `(z, fct_z)`.

    Parameters
    ----------
    z, fct_z : (N_z) complex np.ndarray
        Variable where function is evaluated and function values.
    degree : int, optional
        The difference of denominator and numerator degree. (default: -1)
        This determines how `fct_z` decays for large `abs(z)`:
        `fct_z → z**degree`. For Green's functions it typically is `-1`, for
        self-energies it typically is `0`.
    weight : (N_z) float np.ndarray, optional
        Weighting of the data points, for a known error `σ` this should be
        `weight = 1./σ`.
    moments : (N) float array_like
        Moments of the high-frequency expansion, where
        `f(z) = moments / z**np.arange(1, N+1)` for large `z`. This only
        affects the calculated `pade.residues`, and constrains them to fulfill
        the `moments`.

    Returns
    -------
    pade : PadeApprox
        Padé analytic continuation parametrized by `pade.zeros`,
        `pade.poles` and `pade.residues`.

    Other Parameters
    ----------------
    vandermond : Callable, optional
        Function giving the Vandermond matrix of the chosen polynomial basis.
        Defaults to simple polynomials.
    rotate : bool or None, optional
        Whether to rotate the coordinate to calculated zeros and poles.
        (Default: rotate if `z` is purely imaginary)
    real_asymp : bool, optional
        Whether to consider only the real part of the asymptote, or treat it
        as complex number. Physically, to asymptote should typically be real.
        (Default: True)

    Examples
    --------
    >>> beta = 100
    >>> iws = gt.matsubara_frequencies(range(512), beta=beta)
    >>> gf_iw = gt.square_gf_z(iws, half_bandwidth=1)
    >>> weight = 1./iws.imag  # put emphasis on low frequencies
    >>> gf_pade = gt.polepade.continuation(iws, gf_iw, weight=weight, moments=[1.])

    Compare the result on the real axis:

    >>> import matplotlib.pyplot as plt
    >>> ww = np.linspace(-1.1, 1.1, num=5000)
    >>> __ = plt.plot(ww, gt.square_dos(ww, half_bandwidth=1))
    >>> __ = plt.plot(ww, -1. / np.pi * gf_pade.eval_zeropole(ww).imag)
    >>> __ = plt.plot(ww, -1. / np.pi * gf_pade.eval_polefct(ww).imag)
    >>> plt.show()

    Investigate the pole structure of the continuation:

    >>> gf_pade.plot()
    >>> plt.show()

    """
    if degree > 0:
        raise ValueError(f"`degree` must be smaller or equal 0 (given: {degree}).")
    if rotate is None:  # meant for Matsubara frequencies on imaginary axis
        rotate = np.allclose(z.real, 0)
    if rotate:  # rotate frequencies
        z = z / 1j
        if np.all(~np.iscomplex(z)):
            z = z.real
    m = number_poles(z, fct_z, degree=degree, weight=weight, vandermond=vandermond)
    LOGGER.info("Number of Poles: %s", m)
    pls = poles(z, fct_z, n=m+degree, m=m, weight=weight, vandermond=vandermond)
    zrs = zeros(z, fct_z, poles=pls, n=m+degree, weight=weight, vandermond=vandermond)
    if rotate:  # rotate back
        z, pls, zrs = 1j * z, 1j * pls, 1j * zrs
    asymp, std = asymptotic(z, fct_z, zeros=zrs, poles=pls, weight=weight)
    LOGGER.info("Asymptotic for z**%s: %s ± %s", degree, asymp, std)
    asymp = asymp.real if real_asymp else asymp
    if degree == 0:  # constant has to be treated separately
        fct_z = fct_z - asymp
    residues, err = residues_ols(z, fct_z, pls, weight=weight, moments=moments)
    LOGGER.info("Sum of residues (z**-1): %s; residual %s", residues.sum(), err)
    return PadeApprox(zeros=zrs, poles=pls, residues=residues, amplitude=asymp)
