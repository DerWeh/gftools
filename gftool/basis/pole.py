r"""Representation using poles and the corresponding residues.

Assuming we have only simple poles Green's functions, we can represent Green's
functions using these poles and their corresponding residues:

.. math::

   g(z) = \sum_j r_j / (z - ϵ_j)

where :math:`ϵ_j` are the poles and :math:`r_j` the corresponding residues.
Self-energies can also be represented by the poles after subtracting the static
part.

The pole representation is closely related to the Padé approximation, as rational
polynomials with numerator degree `N` bigger then dominator degree `M`, can also
be represented using `M` poles.

"""
from typing import NamedTuple

import numpy as np
from numpy import newaxis

from gftool import linalg
from gftool.statistics import fermi_fct
from gftool._util import _gu_sum


polyvander = np.polynomial.polynomial.polyvander


def _get_otype(*args):
    """Determine the resulting type if arrays are broadcasted."""
    return sum(np.asarray(arg).reshape(-1)[:1] for arg in args).dtype


def _chebyshev_points(num: int) -> np.ndarray:
    """Return `num` Chebyshev points."""
    return np.sin(0.5 * np.pi / num * np.arange(-num+1, num+1, 2))


class PoleFct(NamedTuple):
    """Function given by finite number of simple `poles` and `residues`.

    Attributes
    ----------
    poles, residues : (..., N) complex np.ndarray
        Poles and residues of the function.

    """

    poles: np.ndarray
    residues: np.ndarray

    def eval_z(self, z):
        """Evaluate the function at `z`."""
        return gf_z(z, poles=self.poles, weights=self.residues)

    def moments(self, order):
        """Calculate high-frequency moments of `order`.

        Parameters
        ----------
        order : (..., M) int array_like
            Order (degree) of the moments. `order` needs to be a positive integer.
            Leading all but the last dimension must be broadcastable with
            `self.poles` and `self.residues`.

        Returns
        -------
        mom : (..., M) float np.ndarray
            High-frequency moments.

        See Also
        --------
        moments

        """
        return moments(poles=self.poles, weights=self.residues, order=order)

    @classmethod
    def from_moments(cls, moments, width=1.):
        """Generate instance matching high-frequency `moments`.

        Parameters
        ----------
        moments : (..., N) float array_like
            Moments of the high-frequency expansion, where
            `g(z) = moments / z**np.arange(1, N+1)` for large `z`.

        Returns
        -------
        PoleFct
            Pole function with high-frequency `moments`.

        See Also
        --------
        gf_from_moments : contains the details how `PoleFct` is constructed.

        """
        return cls(*gf_from_moments(moments, width=width))

    @classmethod
    def from_z(cls, z, gf_z, n_pole, moments=(), width=1., weight=None):
        """Generate instance fitting `gf_z`.

        This function is only meaningful away from the real axis.
        Finds poles and weights for a pole Green's function matching the given
        Green's function `gf_z`.

        Note that for an odd number of moments, the central pole is at `z = 0`,
        so the causal Green's function `g(0)` diverges.

        Parameters
        ----------
        z : (..., N_z) complex np.ndarray
            Frequencies at which `gf_z` is given. Mind that the fit is only
            meaningful away from the real axis.
        gf_z : (..., N_z) complex np.ndarray
            Causal Green's function which is fitted
        n_pole : int
            Number of poles to fit.
        moments : (..., N) float array_like
            Moments of the high-frequency expansion, where
            `G(z) = moments / z**np.arange(N)` for large `z`.
        width : float, optional
            Distance of the largest pole to the origin. (default: 1.)
        weight : (..., N_z) float np.ndarray, optional
            Weighting of the fit. If an error `σ` of the input `gf_z` is known,
            this should be `weight=1/σ`. If high-frequency moments should be fitted
            correctly, `width=abs(z)**(N+1)` is a good fit.

        Returns
        -------
        PoleFct
            Instance with (N) poles at the Chebyshev nodes for degree `N` and
            (..., N) residues such that the pole function fits `gf_z`

        Raises
        ------
        ValueError
            If more moments are given than poles are fitted (`len(moments) > n_pole`)

        See Also
        --------
        gf_from_z

        Notes
        -----
        We employ the similarity of the relation betweens the `moments` and
        the poles and residues with polynomials and the Vandermond matrix.
        The poles are chooses as Chebyshev nodes, the residues are calculated
        accordingly.

        """
        return cls(*gf_from_z(z, gf_z, n_pole=n_pole, moments=moments,
                              width=width, weight=weight))


class PoleGf(PoleFct):
    """Fermionic Green's function given by finite number of `poles` and `residues`."""

    def eval_tau(self, tau, beta):
        """Evaluate the imaginary time Green's function.

        Parameters
        ----------
        tau : (...) float array_like
            Green's function is evaluated at imaginary times `tau`.
            Only implemented for :math:`τ ∈ [0, β]`.
        beta : float
            Inverse temperature

        Returns
        -------
        gf_tau : (...) float np.ndarray
            Imaginary time Green's function.

        See Also
        --------
        gf_tau

        """
        return gf_tau(tau, poles=self.poles, weights=self.residues, beta=beta)

    def eval_ret_t(self, tt):
        """Evaluate the retarded time Green's function.

        Parameters
        ----------
        tt : (...) float array_like
            Green's function is evaluated at times `tt`, for `tt<0` it is 0.

        Returns
        -------
        pole_gf_ret_t : (...) float np.ndarray
            Retarded time Green's function.

        See Also
        --------
        gf_ret_t

        """
        return gf_ret_t(tt, poles=self.poles, weights=self.residues)

    def occ(self, beta):
        """Calculate the occupation number.

        Parameters
        ----------
        beta : float or (..., 1) float array_like
            The inverse temperature :math:`beta = 1/k_B T`.

        Returns
        -------
        occ : (...) float np.ndarray
            Occupation number.

        """
        return _gu_sum(self.residues*fermi_fct(self.poles, beta=beta))

    @classmethod
    def from_tau(cls, gf_tau, n_pole, beta, moments=(), occ=False, width=1., weight=None):
        """Generate instance fitting `gf_tau`.

        Finds poles and weights for a pole Green's function matching the given
        Green's function `gf_tau`.

        Note that for an odd number of moments, the central pole is at `z = 0`,
        so the causal Green's function `g(0)` diverges.

        Parameters
        ----------
        gf_tau : (..., N_tau) float np.ndarray
            Imaginary times Green's function which is fitted.
        n_pole : int
            Number of poles to fit.
        beta : float
            The inverse temperature :math:`beta = 1/k_B T`.
        moments : (..., N) float array_like
            Moments of the high-frequency expansion, where
            `G(z) = moments / z**np.arange(N)` for large `z`.
        width : float, optional
            Distance of the largest pole to the origin. (default: 1.)

        Returns
        -------
        PoleFct
            Instance with (N) poles at the Chebyshev nodes for degree `N` and
            (..., N) residues such that the pole function fits `gf_z`

        Raises
        ------
        ValueError
            If more moments are given than poles are fitted (`len(moments) > n_pole`)

        See Also
        --------
        gf_from_tau

        Notes
        -----
        We employ the similarity of the relation betweens the `moments` and
        the poles and residues with polynomials and the Vandermond matrix.
        The poles are chooses as Chebyshev nodes, the residues are calculated
        accordingly.

        """
        return cls(*gf_from_tau(gf_tau, n_pole=n_pole, beta=beta,
                                moments=moments, occ=occ, width=width, weight=weight))


def gf_z(z, poles, weights):
    """Green's function given by a finite number of `poles`.

    To be a Green's function, `np.sum(weights)` has to be 1 for the `1/z` tail
    or respectively the normalization.

    Parameters
    ----------
    z : (...) complex array_like
        Green's function is evaluated at complex frequency `z`.
    poles, weights : (..., N) float array_like or float
        The position and weight of the poles.

    Returns
    -------
    gf_z : (...) complex np.ndarray
        Green's function.

    See Also
    --------
    gf_d1_z : First derivative of the Green's function
    gf_tau : corresponding fermionic imaginary time Green's function
    gt.pole_gf_tau_b : corresponding bosonic imaginary time Green's function

    """
    poles = np.atleast_1d(poles)
    z = np.asanyarray(z)[..., newaxis]
    return _gu_sum(weights / (z - poles))


_gf_z = gf_z  # keep name, as gf_z is often locally overwritten


def gf_d1_z(z, poles, weights):
    """First derivative of Green's function given by a finite number of `poles`.

    To be a Green's function, `np.sum(weights)` has to be 1 for the 1/z tail.

    Parameters
    ----------
    z : (...) complex array_like
        Green's function is evaluated at complex frequency `z`.
    poles, weights : (..., N) float array_like or float
        The position and weight of the poles.

    Returns
    -------
    gf_d1_z : (...) complex np.ndarray
        Derivative of the Green's function.

    See Also
    --------
    gf_z

    """
    poles = np.atleast_1d(poles)
    z = np.asanyarray(z)[..., newaxis]
    return -_gu_sum(weights * (z - poles)**-2)


def _single_pole_gf_tau(tau, pole, beta):
    assert np.all((tau >= 0.) & (tau <= beta))
    # exp(-tau*pole)*f(-pole, beta) = exp((beta-tau)*pole)*f(pole, beta)
    exponent = np.where(pole.real >= 0, -tau, -tau + beta) * pole
    # -(1-fermi_fct(poles, beta=beta))*np.exp(-tau*poles)
    return -np.exp(exponent) * fermi_fct(-np.sign(pole.real)*pole, beta)


def gf_tau(tau, poles, weights, beta):
    """Imaginary time Green's function given by a finite number of `poles`.

    Parameters
    ----------
    tau : (...) float array_like
        Green's function is evaluated at imaginary times `tau`.
        Only implemented for :math:`τ ∈ [0, β]`.
    poles, weights : (..., N) float array_like or float
        Position and weight of the poles.
    beta : float
        Inverse temperature

    Returns
    -------
    pole_gf_tau : (...) float np.ndarray
        Imaginary time Green's function.

    See Also
    --------
    pole_gf_z : corresponding commutator Green's function

    """
    assert np.all((tau >= 0.) & (tau <= beta))
    poles = np.atleast_1d(poles)
    tau = np.asanyarray(tau)[..., newaxis]
    beta = np.asanyarray(beta)[..., newaxis]
    return _gu_sum(weights*_single_pole_gf_tau(tau, poles, beta=beta))


def _single_pole_gf_ret_t(tt, pole):
    """Retarded time Green's function for a single `pole`."""
    return np.where(tt >= 0, -1j*np.exp(-1j*pole*tt, where=(tt >= 0)), 0)


def gf_ret_t(tt, poles, weights):
    """Retarded time Green's function given by a finite number of `poles`.

    Parameters
    ----------
    tt : (...) float array_like
        Green's function is evaluated at times `tt`, for `tt<0` it is 0.
    poles, weights : (..., N) float array_like or float
        Position and weight of the poles.

    Returns
    -------
    pole_gf_ret_t : (...) float np.ndarray
        Retarded time Green's function.

    See Also
    --------
    pole_gf_z : corresponding commutator Green's function

    """
    poles = np.atleast_1d(poles)
    tt = np.asanyarray(tt)
    return _gu_sum(weights*_single_pole_gf_ret_t(tt[..., newaxis], pole=poles))


def _single_pole_gf_gr_t(tt, pole, beta):
    """Greater time Green's function for a single `pole`."""
    return -1j * np.exp(-1j*pole*tt) * (1 - fermi_fct(pole, beta))


def _single_pole_gf_le_t(tt, pole, beta):
    """Lesser time Green's function for a single `pole`."""
    return 1j * np.exp(-1j*pole*tt) * fermi_fct(pole, beta)


def moments(poles, weights, order):
    r"""High-frequency moments of the pole Green's function.

    Return the moments `mom` of the expansion :math:`g(z) = \sum_m mom_m/z^m`
    For the pole Green's function we have the simple relation
    :math:`1/(z - ϵ) = \sum_{m=1} ϵ^{m-1}/z^m`.

    Parameters
    ----------
    poles, weights : (..., N) float np.ndarray
        Position and weight of the poles.
    order : (..., M) int array_like
        Order (degree) of the moments. `order` needs to be a positive integer.

    Returns
    -------
    mom : (..., M) float np.ndarray
        High-frequency moments.

    """
    poles, weights = np.atleast_1d(*np.broadcast_arrays(poles, weights))
    order = np.asarray(order)[..., newaxis]
    return _gu_sum(weights[..., newaxis, :] * poles[..., newaxis, :]**(order-1))


def gf_from_moments(moments, width=1.) -> PoleFct:
    """Find pole Green's function matching given `moments`.

    Finds poles and weights for a pole Green's function matching the given
    high frequency `moments` for large `z`:
    `g(z) = np.sum(weights / (z - poles)) = moments / z**np.arange(N)`

    Note that for an odd number of moments, the central pole is at `z = 0`,
    so `g(0)` diverges.

    Parameters
    ----------
    moments : (..., N) float array_like
        Moments of the high-frequency expansion, where
        `G(z) = moments / z**np.arange(1, N+1)` for large `z`.
    width : float or (...) float array_like, optional
        Spread of the poles; they are in the interval [-width, width].
        `width=1` are the normal Chebyshev nodes in the interval [-1, 1].
        The default is such, that if the second moment `moments[..., 1]` is
        given, it will be chosen as the largest poles, unless it is small
        (`abs(moments[..., 1]) < 0.1`), then we choose `width=1`.

    Returns
    -------
    gf.resids : (..., N) float np.ndarray
        Residues (or weight) of the poles.
    gf.poles : (N) or (..., N) float np.ndarray
        Position of the poles, these are the Chebyshev nodes for degree `N`.

    Notes
    -----
    We employ the similarity of the relation betweens the `moments` and
    the poles and residues with polynomials and the Vandermond matrix.
    The poles are chooses as Chebyshev nodes, the residues are calculated
    accordingly.

    """
    moments = np.asarray(moments)
    n_mom = moments.shape[-1]
    if n_mom == 0:  # non-sense case, but return consistent behavior
        return PoleFct(poles=np.array([]), residues=moments.copy())
    poles = _chebyshev_points(n_mom)
    if width is None:
        if n_mom <= 1:
            width = 1
        else:  # set width such that second moment is pole unless its very small
            width = np.where(abs(moments[..., 1:2]) >= 0.1,  # arbitrarily chosen threshold
                             abs(moments[..., 1:2])/max(poles), 1)
    poles = width * poles
    _poles, moments = np.broadcast_arrays(poles, moments)
    mat = np.swapaxes(np.polynomial.polynomial.polyvander(_poles, deg=poles.shape[-1]-1), -1, -2)
    resid = np.linalg.solve(mat, moments)
    return PoleFct(poles=poles, residues=resid)


def gf_from_z(z, gf_z, n_pole, moments=(), width=1., weight=None) -> PoleFct:
    """Find pole causal Green's function fitting `gf_z`.

    This function is only meaningful away from the real axis.
    Finds poles and weights for a pole Green's function matching the given
    Green's function `gf_z`.

    Note that for an odd number of moments, the central pole is at `z = 0`,
    so the causal Green's function `g(0)` diverges.

    Parameters
    ----------
    z : (..., N_z) complex np.ndarray
        Frequencies at which `gf_z` is given. Mind that the fit is only
        meaningful away from the real axis.
    gf_z : (..., N_z) complex np.ndarray
        Causal Green's function which is fitted
    n_pole : int
        Number of poles to fit.
    moments : (..., N) float array_like
        Moments of the high-frequency expansion, where
        `G(z) = moments / z**np.arange(N)` for large `z`.
    width : float or None, optional
        Spread of the poles; they are in the interval [-width, width]. (default: 1.)
        `width=1` are the normal Chebyshev nodes in the interval [-1, 1].
        If `width=None` and the second moment `moments[..., 1]` is given,
        it will be chosen as the largest poles, unless it is small
        (`abs(moments[..., 1]) < 0.1`), then we choose `width=1`.
    weight : (..., N_z) float np.ndarray, optional
        Weighting of the fit. If an error `σ` of the input `gf_z` is known,
        this should be `weight=1/σ`. If high-frequency moments should be fitted
        correctly, `weight=abs(z)**(N+1)` is a good fit.

    Returns
    -------
    gf.resids : (..., N) float np.ndarray
        Residues (or weight) of the poles.
    gf.poles : (N) or (..., N) float np.ndarray
        Position of the poles, these are the Chebyshev nodes for degree `N`.

    Raises
    ------
    ValueError
        If more moments are given than poles are fitted (`len(moments) > n_pole`)

    Notes
    -----
    We employ the similarity of the relation betweens the `moments` and
    the poles and residues with polynomials and the Vandermond matrix.
    The poles are chooses as Chebyshev nodes, the residues are calculated
    accordingly.

    """
    moments = np.asarray(moments)
    poles = _chebyshev_points(n_pole)
    if width is None:
        if moments.shape[-1] <= 1:
            width = 1
        else:  # set width such that second moment is pole unless its very small
            width = np.where(abs(moments[..., 1:2]) >= 0.1,  # arbitrarily chosen threshold
                             abs(moments[..., 1:2])/max(poles), 1)
    poles = width * poles
    # z -> newaxis for poles, which are axis=-1
    # poles -> newaxis for sum over axis=-1, newaxis for z which should be axis=-2
    gf_sp_mat = _gf_z(z[..., newaxis], poles[..., newaxis, :, newaxis], weights=1)
    gf_sp_mat = np.concatenate([gf_sp_mat.real, gf_sp_mat.imag], axis=-2)
    gf_z = np.concatenate([gf_z.real, gf_z.imag], axis=-1)
    otype = _get_otype(gf_z, moments, poles)
    if weight is not None:
        weight = np.concatenate([weight, weight], axis=-1)
        gf_sp_mat *= weight[..., np.newaxis]
        gf_z = gf_z * weight
    if moments.shape[-1] > 0:
        if moments.shape[-1] > n_pole:
            raise ValueError("Too many poles given, system is over constrained. "
                             f"poles: {n_pole}, moments: {moments.shape[-1]}")
        constrain_mat = np.swapaxes(polyvander(poles, deg=moments.shape[-1]-1), -1, -2)
        _lstsq_ec = np.vectorize(linalg.lstsq_ec, signature='(m,n),(m),(l,n),(l)->(n)',
                                 otypes=[otype], excluded={'rcond'})
        resid = _lstsq_ec(gf_sp_mat, gf_z, constrain_mat, moments)
    else:
        _lstsq = np.vectorize(lambda a, b: np.linalg.lstsq(a, b, rcond=None)[0],
                              signature='(m,n),(m)->(n)', otypes=[otype])
        resid = _lstsq(gf_sp_mat, gf_z)
    return PoleFct(poles=poles, residues=resid)


def gf_from_tau(gf_tau, n_pole, beta, moments=(), occ=False, width=1., weight=None) -> PoleGf:
    """Find pole Green's function fitting `gf_tau`.

    Finds poles and weights for a pole Green's function matching the given
    Green's function `gf_tau`.

    Note that for an odd number of moments, the central pole is at `z = 0`,
    so the causal Green's function `g(0)` diverges.

    Parameters
    ----------
    gf_tau : (..., N_tau) float np.ndarray
        Imaginary times Green's function which is fitted.
    n_pole : int
        Number of poles to fit.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.
    moments : (..., N) float array_like
        Moments of the high-frequency expansion, where
        `G(z) = moments / z**np.arange(N)` for large `z`.
    width : float, optional
        Distance of the largest pole to the origin. (default: 1.)

    Returns
    -------
    gf.resids : (..., N) float np.ndarray
        Residues (or weight) of the poles.
    gf.poles : (N) float np.ndarray
        Position of the poles, these are the Chebyshev nodes for degree `N`.

    Raises
    ------
    ValueError
        If more moments are given than poles are fitted (`len(moments) > n_pole`)

    Notes
    -----
    We employ the similarity of the relation betweens the `moments` and
    the poles and residues with polynomials and the Vandermond matrix.
    The poles are chooses as Chebyshev nodes, the residues are calculated
    accordingly.

    """
    poles = width * _chebyshev_points(n_pole)
    tau = np.linspace(0, beta, num=gf_tau.shape[-1])
    gf_sp_mat = _single_pole_gf_tau(tau[..., newaxis], poles, beta=beta)
    moments = np.asarray(moments)
    otype = _get_otype(gf_tau, moments, poles)
    if weight is not None:
        gf_sp_mat *= weight[..., np.newaxis]
        gf_tau = gf_tau * weight
    if moments.shape[-1] > 0 or occ:  # constrained
        if moments.shape[-1] + int(occ) > n_pole:
            raise ValueError("Too many poles given, system is over constrained. "
                             f"poles: {n_pole}, moments: {moments.shape[-1]}")
        constrain_mat = np.polynomial.polynomial.polyvander(poles, deg=moments.shape[-1]-1).T
        if occ:
            constrain_mat = np.concatenate(
                np.broadcast_arrays(constrain_mat, fermi_fct(poles, beta=beta)), axis=-2
            )
            moments = np.concatenate(np.broadcast_arrays(moments, occ), axis=-1)
        _lstsq_ec = np.vectorize(linalg.lstsq_ec, signature='(m,n),(m),(l,n),(l)->(n)',
                                 otypes=[otype], excluded={'rcond'})
        resid = _lstsq_ec(gf_sp_mat, gf_tau, constrain_mat, moments)
    else:
        _lstsq = np.vectorize(lambda a, b: np.linalg.lstsq(a, b, rcond=None)[0],
                              signature='(m,n),(m)->(n)', otypes=[otype])
        resid = _lstsq(gf_sp_mat, gf_tau)
    return PoleGf(poles=poles, residues=resid)
