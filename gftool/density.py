"""Calculate density from Green's function."""
import numpy as np

from gftool._util import _gu_sum
from gftool.basis.pole import PoleGf


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
