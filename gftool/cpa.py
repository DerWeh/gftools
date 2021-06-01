"""Coherent cluster approximation (CPA) to substitutional disorder.

For a high-level interface use `solve_root` to solve the CPA problem for
arbitrary frequencies `z`.
Solutions for fixed occupation `occ` can be obtained on the imaginary axis only
using `solve_fxdocc_root`.

Fixed occupation on the real axis is currently not support. We recommend
obtaining the chemical potential `mu` for the given `occ` using
`solve_fxdocc_root` on the imaginary axis, and then run `solve_root` with the
given `mu` on the real axis. In fact, we expect this to be more stable than
fixing the charge on the real axis directly.

"""
# pylint: disable=too-many-locals
import logging

from functools import partial
from typing import Callable, NamedTuple

import numpy as np

from scipy import optimize

from gftool.density import density_iw, chemical_potential


LOGGER = logging.getLogger(__name__)


def _join(*args):
    """Join arguments to 1D array for use in `optimize`.

    Parameters
    ----------
    args : array_like
        Arrays to join together.

    Returns
    -------
    joined : np.ndarray
        1D array containing all flattened elements of `args`.
    shapes : list of tuple of int
        Shapes of all `args`. This is necessary to perform the inverse
        operation `_split`.

    """
    args = [np.asanyarray(ar) for ar in args]
    joined = np.concatenate([ar.reshape(-1) for ar in args])
    shapes = [ar.shape for ar in args]
    return joined, shapes


def _split(joined, shapes):
    """Inverse operation to `_join` separating arrays."""
    sizes = [np.prod(sh) for sh in shapes[:-1]]
    splited = np.split(joined, indices_or_sections=np.cumsum(sizes))
    return [array.reshape(sh) for array, sh in zip(splited, shapes)]


def gf_cmpt_z(z, self_cpa_z, e_onsite, hilbert_trafo: Callable[[complex], complex]):
    """Green's function for the components embedded in `self_cp_z`.

    Parameters
    ----------
    z, self_cp_z : (...) complex np.ndarray
        Frequency points and corresponding CPA self-energy.
    e_onsite : (..., N_cmpt) float of complex np.ndarray
        On-site energy of the components. This can also include a local
        frequency dependent self-energy of the component sites.
    hilbert_trafo : Callable[[complex], complex]
        Hilbert transformation of the lattice to calculate the coherent Green's
        function.

    Returns
    -------
    gf_cmpt_z : (..., N_cmpt) complex np.ndarray
        The Green's function of the components embedded in `self_cpa_z`.

    """
    gf_coher_z = hilbert_trafo(z - self_cpa_z)[..., np.newaxis]
    return gf_coher_z / (1 - (e_onsite - self_cpa_z[..., np.newaxis])*gf_coher_z)


def self_root_eq(self_cpa_z, z, e_onsite, concentration,
                 hilbert_trafo: Callable[[complex], complex]):
    """Root equation r(Σ)=0 for CPA.

    The root equation writes `r(Σ, z) = T(z) / (1 + T(z)*hilbert_trafo(z-Σ))`.

    Parameters
    ----------
    self_cp_z : (...) complex np.ndarray
        CPA self-energy.
    z : (...) complex array_like
        Frequency points.
    e_onsite : (..., N_cmpt) float or complex np.ndarray
        On-site energy of the components. This can also include a local
        frequency dependent self-energy of the component sites.
    concentration : (..., N_cmpt) float array_like
        Concentration of the different components used for the average.
    hilbert_trafo : Callable[[complex], complex]
        Hilbert transformation of the lattice to calculate the coherent Green's
        function.

    Returns
    -------
    remainder : (...) complex np.ndarray
        The result of r(Σ), if it is `0` and hence a root, `self_cp_z` is the
        correct CPA self-energy.

    """
    gf_coher_z = hilbert_trafo(z - self_cpa_z)
    energy_diff = e_onsite - self_cpa_z[..., np.newaxis]
    T = np.sum(concentration * energy_diff / (1 - energy_diff*gf_coher_z[..., np.newaxis]), axis=-1)
    return T/(1+T*gf_coher_z)


def self_fxdpnt_eq(self_cpa_z, z, e_onsite, concentration,
                   hilbert_trafo: Callable[[complex], complex]):
    """Fixed-point equation f(Σ)=Σ for CPA.

    The fixed-point equation writes `f(Σ, z) = Σ + T(z) / (1 + T(z)*hilbert_trafo(z-Σ))`.

    Parameters
    ----------
    self_cp_z : (...) complex np.ndarray
        CPA self-energy.
    z : (...) complex array_like
        Frequency points.
    e_onsite : (..., N_cmpt) float complex np.ndarray
        On-site energy of the components. This can also include a local
        frequency dependent self-energy of the component sites.
    concentration : (..., N_cmpt) float array_like
        Concentration of the different components used for the average.
    hilbert_trafo : Callable[[complex], complex]
        Hilbert transformation of the lattice to calculate the coherent Green's
        function.

    Returns
    -------
    self_cpa_z_new : (..., N_z) complex np.ndarray
        The new self-energy f(Σ), if it is Σ again and hence a fixed-point,
        `self_cpa_z_new` is the correct CPA self-energy.

    """
    return self_cpa_z + self_root_eq(self_cpa_z, z, e_onsite=e_onsite, concentration=concentration,
                                     hilbert_trafo=hilbert_trafo)


def restrict_self_root_eq(self_cpa_z, *args, **kwds):
    """Wrap `self_root_eq` to restrict the solutions to `self_cpa_z.imag > 0`."""
    unphysical = self_cpa_z.imag > 0
    if np.all(~unphysical):  # no need for restrictions
        return self_root_eq(self_cpa_z, *args, **kwds)
    distance = self_cpa_z.imag[unphysical].copy()  # distance to physical solution
    # print('>', max(distance))
    self_cpa_z.imag[unphysical] = 0
    root = np.asanyarray(self_root_eq(self_cpa_z, *args, **kwds))
    root[unphysical] *= (1 + distance)  # linearly enlarge residues
    # kill unphysical roots
    root.real[unphysical] += 1e-3 * distance * np.where(root.real[unphysical] >= 0, 1, -1)
    root.imag[unphysical] += 1e-3 * distance * np.where(root.imag[unphysical] >= 0, 1, -1)
    return root


def solve_root(z, e_onsite, concentration, hilbert_trafo: Callable[[complex], complex],
               self_cpa_z0=None, restricted=True, **root_kwds):
    """Determine the CPA self-energy by solving the root problem.

    Note that the result should be checked, whether the obtained solution is
    physical.

    Parameters
    ----------
    z : (...) complex array_like
        Frequency points.
    e_onsite : (..., N_cmpt) float or complex np.ndarray
        On-site energy of the components. This can also include a local
        frequency dependent self-energy of the component sites.
    concentration : (..., N_cmpt) float array_like
        Concentration of the different components used for the average.
    hilbert_trafo : Callable[[complex], complex]
        Hilbert transformation of the lattice to calculate the coherent Green's
        function.
    self_cpa_z0 : (...) complex np.ndarray, optional
        Starting guess for CPA self-energy.
    restricted : bool, optional
        Whether `self_cpa_z` is restricted to `self_cpa_z.imag <= 0`. (default: True)
        Note, that even if `restricted=True`, the imaginary part can get negative
        within tolerance. This should be removed by hand if necessary.
    root_kwds
        Additional arguments passed to `scipy.optimize.root`.
        `method` can be used to choose a solver.
        `options=dict(fatol=tol)` can be specified to set the desired tolerance
        `tol`.

    Returns
    -------
    self_cpa_z : (...) complex np.ndarray
        The CPA self-energy as the root of `self_root_eq`.

    Raises
    ------
    RuntimeError
        If unable to find a solution.

    Examples
    --------
    >>> from functools import partial
    >>> parameter = dict(
    ...     e_onsite=[-0.3, 0.3],
    ...     concentration=[0.3, 0.7],
    ...     hilbert_trafo=partial(gt.bethe_gf_z, half_bandwidth=1),
    ... )

    >>> ww = np.linspace(-1.5, 1.5, num=5000) + 1e-10j
    >>> self_cpa_ww = gt.cpa.solve_root(ww, **parameter)
    >>> del parameter['concentration']
    >>> gf_cmpt_ww = gt.cpa.gf_cmpt_z(ww, self_cpa_ww, **parameter)

    >>> import matplotlib.pyplot as plt
    >>> __ = plt.plot(ww.real, -1./np.pi*gf_cmpt_ww[..., 0].imag)
    >>> __ = plt.plot(ww.real, -1./np.pi*gf_cmpt_ww[..., 1].imag)
    >>> plt.show()

    Notes
    -----
    For `restricted=True` root-serach, we made good experince with the methods
    `'anderson'`, `'krylov'` and `'df-sane'`.
    For `restricted=False`, we made made good experince with the method `'broyden2'`.

    """
    concentration = np.array(concentration)
    if self_cpa_z0 is None:  # static average + 0j to make it complex array
        self_cpa_z0 = np.sum(e_onsite * concentration, axis=-1) + 0j
        self_cpa_z0, __ = np.broadcast_arrays(self_cpa_z0, z)
    else:  # make sure that `self_cpa_z0` has right shape of output for root
        output = np.broadcast(z, e_onsite[..., 0], concentration[..., 0], self_cpa_z0)
        self_cpa_z0 = np.broadcast_to(self_cpa_z0, shape=output.shape)
    root_eq = partial(restrict_self_root_eq if restricted else self_root_eq,
                      z=z, e_onsite=e_onsite, concentration=concentration,
                      hilbert_trafo=hilbert_trafo)

    root_kwds.setdefault("method", "anderson" if restricted else "broyden2")
    sol = optimize.root(root_eq, x0=self_cpa_z0, **root_kwds)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.x


class RootFxdocc(NamedTuple):
    """CPA solution for the self-energy root-equation for fixed occupation.

    Attributes
    ----------
    self_cpa : np.ndarray or complex
        The CPA self-energy.
    mu : float
        Chemical potential.

    """

    self_cpa: np.ndarray
    mu: float


def solve_fxdocc_root(iws, e_onsite, concentration, hilbert_trafo: Callable[[complex], complex],
                      beta: float, occ: float = None, self_cpa_iw0=None, mu0: float = 0,
                      weights=1, n_fit=0, restricted=True, **root_kwds) -> RootFxdocc:
    """Determine the CPA self-energy by solving the root problem for fixed `occ`.

    Parameters
    ----------
    iws : (N_iw) complex array_like
        Positive fermionic Matsubara frequencies.
    e_onsite : (N_cmpt) float or (..., N_iw, N_cmpt) complex np.ndarray
        On-site energy of the components. This can also include a local
        frequency dependent self-energy of the component sites.
        If multiple non-frequency dependent on-site energies should be
        considered simultaneously, pass an on-site energy with `N_z=1`:
        `e_onsite[..., np.newaxis, :]`.
    concentration : (..., N_cmpt) float array_like
        Concentration of the different components used for the average.
    hilbert_trafo : Callable[[complex], complex]
        Hilbert transformation of the lattice to calculate the coherent Green's
        function.
    beta : float
        Inverse temperature.
    occ : float
        Total occupation.
    self_cpa_iw0, mu0 : (..., N_iw) complex np.ndarray and float, optional
        Starting guess for CPA self-energy and chemical potential.
        `self_cpa_iw0` implicitly contains the chemical potential `mu0`,
        thus they should match.

    Returns
    -------
    root.self_cpa : (..., N_iw) complex np.ndarray
        The CPA self-energy as the root of `self_root_eq`.
    root.mu : float
        Chemical potential for the given occupation `occ`.

    Other Parameters
    ----------------
    weights : (N_iw) float np.ndarray, optional
        Passed to `gftool.density_iw`.
        Residues of the frequencies with respect to the residues of the
        Matsubara frequencies `1/beta`. (default: 1.)
        For Padé frequencies this needs to be provided.
    n_fit : int, optional
        Passed to `gftool.density_iw`.
        Number of additionally fitted moments. If Padé frequencies
        are used, this is typically not necessary. (default: 0)
    restricted : bool, optional
        Whether `self_cpa_z` is restricted to `self_cpa_z.imag <= 0`. (default: True)
        Note, that even if `restricted=True`, the imaginary part can get negative
        within tolerance. This should be removed by hand if necessary.
    root_kwds
        Additional arguments passed to `scipy.optimize.root`.
        `method` can be used to choose a solver.
        `options=dict(fatol=tol)` can be specified to set the desired tolerance
        `tol`.

    Raises
    ------
    RuntimeError
        If unable to find a solution.

    See Also
    --------
    solve_root

    Examples
    --------
    >>> from functools import partial
    >>> beta = 30
    >>> e_onsite = [-0.3, 0.3]
    >>> conc = [0.3, 0.7]
    >>> hilbert = partial(gt.bethe_gf_z, half_bandwidth=1)
    >>> occ = 0.5,

    >>> iws = gt.matsubara_frequencies(range(1024), beta=30)
    >>> self_cpa_iw, mu = gt.cpa.solve_fxdocc_root(iws, e_onsite, conc,
    ...                                            hilbert, occ=occ, beta=beta)

    >>> import matplotlib.pyplot as plt
    >>> __ = plt.plot(iws.imag, self_cpa_iw.imag, '+--')
    >>> __ = plt.axhline(np.average(e_onsite, weights=conc) - mu)
    >>> __ = plt.plot(iws.imag, self_cpa_iw.real, 'x--')
    >>> plt.show()

    check occupation

    >>> gf_coher_iw = hilbert(iws - self_cpa_iw)
    >>> gt.density_iw(iws, gf_coher_iw, beta=beta, moments=[1, self_cpa_iw[-1].real])
    0.499999...

    check CPA

    >>> self_compare = gt.cpa.solve_root(iws, np.array(e_onsite)-mu, conc,
    ...                                  hilbert_trafo=hilbert)
    >>> np.allclose(self_cpa_iw, self_compare, atol=1e-5)
    True

    """
    concentration = np.asarray(concentration)[..., np.newaxis, :]
    e_onsite = np.asarray(e_onsite)
    if self_cpa_iw0 is None:  # static average + 0j to make it complex array
        self_cpa_iw0 = np.sum(e_onsite * concentration, axis=-1) - mu0 + 0j
        self_cpa_iw0, __ = np.broadcast_arrays(self_cpa_iw0, iws)
    self_cpa_nomu = self_cpa_iw0 + mu0  # strip contribution of mu

    # TODO: use on-site energy to estimate m2+mu, which only has to be adjusted by mu
    m1 = np.ones_like(self_cpa_iw0[..., -1].real)

    def _occ_diff(x):
        gf_coher_iw = hilbert_trafo(iws - x)
        m2 = x[..., -1].real  # for large iws, real part should static part
        occ_root = density_iw(iws, gf_iw=gf_coher_iw, beta=beta, weights=weights,
                              moments=np.stack([m1, m2], axis=-1), n_fit=n_fit).sum()
        return occ_root - occ

    mu = chemical_potential(lambda mu: _occ_diff(self_cpa_nomu - mu), mu0=mu0)
    LOGGER.debug("VCA chemical potential: %s", mu)
    # one iteration gives the ATA: average t-matrix approximation
    self_cpa_nomu = self_fxdpnt_eq(self_cpa_nomu - mu, iws, e_onsite - mu,
                                   concentration, hilbert_trafo) + mu
    mu = chemical_potential(lambda mu: _occ_diff(self_cpa_nomu - mu), mu0=mu)
    LOGGER.debug("ATA chemical potential: %s", mu)

    x0, shapes = _join([mu], self_cpa_nomu.real, self_cpa_nomu.imag)
    self_root_eq_ = partial(restrict_self_root_eq if restricted else self_root_eq,
                            z=iws, concentration=concentration, hilbert_trafo=hilbert_trafo)

    def root_eq(mu_selfcpa):
        mu, self_cpa_re, self_cpa_im = _split(mu_selfcpa, shapes)
        self_cpa = self_cpa_re + 1j*self_cpa_im - mu  # add contribution of mu
        self_root = self_root_eq_(self_cpa, e_onsite=e_onsite - mu)
        occ_root = _occ_diff(self_cpa)
        return _join([self_root.size*occ_root],
                     self_root.real, self_root.imag)[0]

    root_kwds.setdefault("method", "krylov")
    LOGGER.debug('Search BEB self-energy root')
    if 'callback' not in root_kwds and LOGGER.isEnabledFor(logging.DEBUG):
        # setup LOGGER if no 'callback' is provided
        root_kwds['callback'] = lambda x, f: LOGGER.debug(
            'Residue: mu=%+6g   cpa=%6g', f[0], np.linalg.norm(f[1:])
        )

    sol = optimize.root(root_eq, x0=x0, **root_kwds)
    LOGGER.debug("CPA self-energy root found after %s iterations.", sol.nit)
    if not sol.success:
        raise RuntimeError(sol.message)
    mu, self_cpa_re, self_cpa_im = _split(sol.x, shapes)
    self_cpa = self_cpa_re - mu + 1j*self_cpa_im  # add contribution of mu
    LOGGER.debug("CPA chemical potential: %s", mu.item())
    return RootFxdocc(self_cpa, mu=mu.item())
