"""Padé based on robust pole finding.

Instead of fitting a rational polynomial, poles and zeros or poles and
corresponding residues are fitted.

The algorithm is based on [ito2018]_ and adjusted to Green's functions and self-
energies. We assume that we know exactly the high-frequency behavior of the
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

"""
import logging

from typing import NamedTuple

import numpy as np

from numpy.polynomial import polynomial
from scipy import linalg

import gftool as gt
from gftool.basis import ZeroPole, PoleFct

LOGGER = logging.getLogger(__name__)


class PadeApprox(NamedTuple):
    """Representation of the Padé approximation based on poles.

    Basically the approximation is obtained as `PoleFct` as well as as `ZeroPole`.
    Not however that those to approximations will in general no agree. For a
    good approximation however, they should be very similar.

    Attributes
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
        import matplotlib as mpl  # pylint: disable=[import-outside-toplevel]
        import matplotlib.pyplot as plt  # pylint: disable=[import-outside-toplevel]
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
    if n is None:
        n = m - 1
    fct_z = fct_z/np.median(fct_z)
    vander = vandermond(z, deg=max(n+1, m))
    numer = -vander[..., :n+1]
    denom = vander[..., :m]
    scaling = 1./np.linalg.norm(np.concatenate((denom, numer), axis=-1), axis=-1, keepdims=True)

    if weight is not None:
        scaling *= weight[..., np.newaxis]
    q_numer, __ = np.linalg.qr(scaling*numer, mode='complete')
    q_fct_denom, __ = np.linalg.qr(scaling*fct_z[..., np.newaxis]*denom, mode='reduced')
    # q_fct_denom, *__ = spla.qr(D*B1, mode='economic', pivoting=True)
    qtilde_z_fct_denom = q_numer[..., n+1:].T.conj() @ (z[..., np.newaxis] * q_fct_denom)
    qtilde_fct_denom = q_numer[..., n+1:].T.conj() @ q_fct_denom
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
    m = poles.size
    if n is None:
        n = m - 1
    fct_z = fct_z/np.median(fct_z)
    denom = np.prod(np.subtract.outer(z, poles), axis=-1)
    numer = vandermond(z, deg=n-1)
    scaling = 1./np.linalg.norm(numer, axis=-1, keepdims=True)  # <- by far best
    scaling = 1./np.linalg.norm(np.concatenate((denom[..., np.newaxis], numer), axis=-1),
                                axis=-1, keepdims=True)
    if weight is not None:
        scaling *= weight[..., np.newaxis]
    q_fct_denom, __ = np.linalg.qr(scaling*(fct_z*denom)[:, np.newaxis], mode='complete')
    q_numer, __ = np.linalg.qr(scaling*numer, mode='reduced')
    qtilde_z_numer = q_fct_denom[:, 1:].T.conj() @ (z[..., np.newaxis] * q_numer)
    qtilde_numer = q_fct_denom[:, 1:].T.conj() @ q_numer
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
    resid = gt.linalg.lstsq_ec(polematrix, fct_z, constrain_mat, moments)
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
    vandermond : Callable, optional
        Function giving the Vandermond matrix of the chosen polynomial basis.
        Defaults to simple polynomials.

    Returns
    -------
    pade : PadeApprox
        Padé analytic continuation parametrized by `pade.zeros`,
        `pade.poles` and `pade.residues`.

    Other Parameters
    ----------------
    rotate : bool, optional
        Whether to rotate the coordinate to calculated zeros and poles.
        (Default: rotate if `z` is purely imaginary)

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
    m = number_poles(z, fct_z, degree=degree, weight=weight, vandermond=vandermond)
    LOGGER.info("Number of Poles: %s", m)
    if rotate is None:
        rotate = np.allclose(z.real, 0)
    pls = poles(z.imag if rotate else z, fct_z, n=m+degree, m=m,
                weight=weight, vandermond=vandermond)
    zrs = zeros(z.imag if rotate else z, fct_z, poles=pls, n=m+degree,
                weight=weight, vandermond=vandermond)
    if rotate:
        pls, zrs = 1j * pls, 1j * zrs
    asymp, std = asymptotic(z, fct_z, zeros=zrs, poles=pls, weight=weight)
    LOGGER.info("Asymptotic for z**%s: %s ± %s", degree, asymp, std)
    asymp = asymp.real if real_asymp else asymp
    const = asymp if degree == 0 else 0
    fct_z_pole = fct_z - const  # constant has to be treated separately
    residues, err = residues_ols(z, fct_z_pole, pls, weight=weight, moments=moments)
    LOGGER.info("Sum of residues (z**-1): %s; residual %s", residues.sum(), err)
    return PadeApprox(zeros=zrs, poles=pls, residues=residues, amplitude=asymp)