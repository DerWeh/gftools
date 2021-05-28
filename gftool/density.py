"""Calculate density from Green's function."""
import logging

from typing import Callable

import numpy as np

from scipy import optimize

from gftool._util import _gu_sum
from gftool.basis.pole import PoleGf

LOGGER = logging.getLogger(__name__)


def density_iw(iws, gf_iw, beta, weights=1., moments=(1.,), n_fit=0):
    r"""Calculate the number density of the Green's function `gf_iw` at finite temperature `beta`.

    This function can be used for fermionic Matsubara frequencies `matsubara_frequencies`,
    as well as fermionic Padé frequencies `pade_frequencies`.

    Parameters
    ----------
    iws, gf_iw : (..., N_iw) complex np.ndarray
        Positive Matsubara frequencies :math:`iω_n` (or Padé :math:`iz_p`)
        and the Green's function at these frequencies.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.
    weights : (..., N_iw) float np.ndarray, optional
        Residues of the frequencies with respect to the residues of the
        Matsubara frequencies `1/beta`. (default: 1.)
        For Padé frequencies this needs to be provided.
    moments : (..., M) float array_like, optional
        Moments of the high-frequency expansion, where
        `G(z) = np.sum(moments / z**np.arange(N))` for large `z`.
    n_fit : int, optional
        Number of additionally to `moments` fitted moments. If Padé frequencies
        are used, this is typically not necessary. (default: 0)

    Returns
    -------
    occ : float
        The number density of the given Green's function `gf_iw`.

    See Also
    --------
    matsubara_frequencies : Method generating Matsubara frequencies `iws`.
    pade_frequencies : Method generating Padé frequencies `iws` with `weights`.

    Examples
    --------
    >>> BETA = 50
    >>> iws = gt.matsubara_frequencies(range(1024), beta=BETA)

    Example Green's function

    >>> np.random.seed(0)  # to have deterministic results
    >>> poles = 2*np.random.random(10) - 1  # partially filled
    >>> residues = np.random.random(10); residues = residues / np.sum(residues)
    >>> pole_gf = gt.basis.PoleGf(poles=poles, residues=residues)
    >>> gf_iw = pole_gf.eval_z(iws)
    >>> exact = pole_gf.occ(BETA)
    >>> exact
    0.17858151698239388

    Numerical calculation of the occupation number,
    using Matsubara frequency

    >>> occ = gt.density_iw(iws, gf_iw, beta=BETA)
    >>> m2 = pole_gf.moments(2)  # additional high-frequency moment
    >>> occ_m2 = gt.density_iw(iws, gf_iw, beta=BETA, moments=[1., m2])
    >>> occ_fit2 = gt.density_iw(iws, gf_iw, beta=BETA, n_fit=1)
    >>> exact, occ, occ_m2, occ_fit2
    (0.17858151..., 0.17934437..., 0.17858150..., 0.17858198...)
    >>> abs(occ - exact), abs(occ_m2 - exact), abs(occ_fit2 - exact)
    (0.00076286..., 8.18...e-09, 4.72...e-07)

    using more accurate Padé frequencies

    >>> izp, rp = gt.pade_frequencies(100, beta=BETA)
    >>> gf_izp = pole_gf.eval_z(izp)
    >>> occ_izp = gt.density_iw(izp, gf_izp, beta=BETA, weights=rp)
    >>> occ_izp
    0.17858151...
    >>> abs(occ_izp - exact) < 1e-14
    True

    """
    # add axis for iws, remove it later at occupation
    moments = np.asanyarray(moments, dtype=np.float_)[..., np.newaxis, :]
    if n_fit:
        n_mom = moments.shape[-1]
        weight = iws.imag**(n_mom+n_fit)
        mom_gf = PoleGf.from_z(iws, gf_iw[..., np.newaxis, :], n_pole=n_fit+n_mom,
                               moments=moments, width=None, weight=weight)
    else:
        mom_gf = PoleGf.from_moments(moments, width=None)
    delta_gf_iw = gf_iw.real - mom_gf.eval_z(iws).real
    return 2./beta*_gu_sum(weights * delta_gf_iw.real) + mom_gf.occ(beta)[..., 0]


def chemical_potential(occ_root: Callable[[float], float], mu0=0.0, step0=1.0, **kwds) -> float:
    """Search chemical potential for a given occupation.

    Parameters
    ----------
    occ_root : callable
        Function `occ_root(mu_i) -> occ_i - occ`, which returns the difference
        in occupation to the desired occupation `occ` for a chemical potential
        `mu_i`.
        Note that the sign is important, `occ_i - occ` has to be returned!
    mu0 : float, optional
        The starting guess for the chemical potential. (default: 0)
    step0 : float, optional
        Starting step-width for the bracket search. A reasonable guess is of
        the order of the band-width. (default: 1)
    kwds
        Additional keyword arguments passed to `scipy.optimize.root_scalar`.
        Common arguments might be `xtol` or `rtol` for absolute or relative
        tolerance.

    Returns
    -------
    mu : float
        The chemical potential given the correct charge `occ_root(mu)=0`.

    Raises
    ------
    RuntimeError
        If either no bracket can be found (this should only happen for the
        complete empty or completely filled case),
        or if the scalar root search in the bracket fails.

    Notes
    -----
    The search for a chemical potential is a two-step procedure:
    *First*, we search for a bracket `[mua, mub]` with
    `occ_root(mua) < 0 < occ_root(mub)`. We use that the occupation is a
    monotonous increasing function of the chemical potential `mu`.
    *Second*, we perform a standard root-search in `[mua, mub]` which is done
    using `scipy.optimize.root_scalar`, Brent's method is currently used as
    default.

    Examples
    --------
    We search for the occupation of a simple 3-level system, where the
    occupation of each level is simply given by the Fermi function:

    >>> occ = 1.67  # desired total occupation
    >>> BETA = 100  # inverse temperature
    >>> eps = np.random.random(3)
    >>> def occ_fct(mu):
    ...     return gt.fermi_fct(eps - mu, beta=BETA).sum()
    >>> mu = gt.chemical_potential(lambda mu: occ_fct(mu) - occ)
    >>> occ_fct(mu), occ
    (1.67000..., 1.67)

    """
    # find a bracket
    delta_occ0 = occ_root(mu0)
    if delta_occ0 == 0:  # has already correct occupation
        return mu0
    sign0 = np.sign(delta_occ0)  # whether occupation is too large or too small
    step = -step0 * delta_occ0

    mu1 = mu0
    loops = 0
    while np.sign(occ_root(mu0 + step)) == sign0:
        mu1 = mu0 + step
        step *= 2  # increase step width exponentially till a bounds are found
        loops += 1
        if loops > 100:
            raise RuntimeError("No bracket `occ_root(mua) < 0 < occ_root(mub)` could be found.")
    bracket = list(sorted([mu1, mu0+step]))
    LOGGER.debug("Bracket found after %s iterations.", loops)
    root_res = optimize.root_scalar(occ_root, bracket=bracket, **kwds)
    if not root_res.converged:
        runtime_err = RuntimeError(
            f"Root-search for chemical potential failed after {root_res.iterations}.\n"
            f"Cause of failure: {root_res.flag}"
        )
        runtime_err.mu = root_res.root
        raise runtime_err
    LOGGER.debug("Root found after %s additional evaluations.", root_res.function_calls)
    return root_res.root
