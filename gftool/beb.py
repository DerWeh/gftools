"""Blackman, Esterling, and Berk (BEB) approach to off-diagonal disorder.

Extension to CPA allowing for random hopping amplitudes. [blackman1971]_

The current implementation is based on QR decomposition instead of generalized
eigendecomposition, as no vectorized version of the eigendecomposition exists.
The eigendecomposition is extremely slow.


References
----------
.. [blackman1971]
   Blackman, J.A., Esterling, D.M., Berk, N.F., 1971.
   Generalized Locator---Coherent-Potential Approach to Binary Alloys.
   Phys. Rev. B 4, 2412–2428. https://doi.org/10.1103/PhysRevB.4.2412

"""
import logging

from typing import Callable
from functools import partial

import numpy as np

from scipy import optimize
from numpy import newaxis

from gftool import matrix

LOGGER = logging.getLogger(__name__)

# gu-function versions to extract diagonal and transpose matrices
diagonal = partial(np.diagonal, axis1=-2, axis2=-1)
transpose = partial(np.swapaxes, axis1=-1, axis2=-2)


def gf_loc_z(z, self_beb_z, hopping, hilbert_trafo: Callable[[complex], complex],
             diagnal=True):
    """Calculate average local Green's function matrix in components.

    For the self-consistent self-energy `self_beb_z` this it is diagonal in the
    components. Not that `gf_loc_z` implicitly contains the concentrations.

    Parameters
    ----------
    z : (..., N_z) complex np.ndarray
        Frequency points.
    self_beb_z : (..., N_z, N_cmpt, N_cmpt) complex np.ndarray
        BEB self-energy.
    hopping : (N_cmpt, N_cmpt) float array_like
        Hopping matrix in the components.
    hilbert_trafo : Callable[[complex], complex]
        Hilbert transformation of the lattice to calculate the local Green's function.
    diagnal : bool, optional
        If `diagnal`, only the diagonal elements are calculated, else the full
        matrix.

    Returns
    -------
    gf_loc_z : (..., N_z, N_cmpt) or (..., N_z, N_cmpt, N_cmpt) complex np.ndarray
        The average local Green's function matrix.

    """
    eye = np.eye(*hopping.shape)
    qt, rt = np.linalg.qr(hopping)
    rt_inv = np.linalg.inv(rt)
    # [..., newaxis]*eye add matrix axis
    z_m_self = z[..., newaxis, newaxis]*eye - self_beb_z
    eig, rv = np.linalg.eig(qt.T @ z_m_self @ rt_inv)
    dec = matrix.Decomposition(qt@rv, eig, np.linalg.inv(rv)@rt)

    return dec.reconstruct(hilbert_trafo(dec.xi), kind='diag' if diagnal else 'full')


def self_root_eq(self_beb_z, z, e_onsite, concentration, hopping,
                 hilbert_trafo: Callable[[complex], complex]):
    """Root equation r(Σ)=0 for BEB.

    Parameters
    ----------
    self_beb_z : (..., N_z, N_cmpt, N_cmpt) complex np.ndarray
        BEB self-energy.
    z : (N_z) complex np.ndarray
        Frequency points.
    e_onsite : (N_cmpt) float or complex array_like
        On-site energy of the components.
    concentration : (N_cmpt) float array_like
        Concentration of the different components.
    hopping : (N_cmpt, N_cmpt) float array_like
        Hopping matrix in the components.
    hilbert_trafo : Callable[[complex], complex]
        Hilbert transformation of the lattice to calculate the local Green's function.

    """
    eye = np.eye(*hopping.shape)
    qt, rt = np.linalg.qr(hopping)
    rt_inv = np.linalg.inv(rt)
    # [..., newaxis]*eye add matrix axis
    # matrix-products are faster if larger arrays are in Fortran order
    z_m_self = np.asfortranarray(z[..., newaxis, newaxis]*eye - self_beb_z)
    eig, rv = np.linalg.eig(qt.T @ z_m_self @ rt_inv)
    rv = np.asfortranarray(rv)
    dec = matrix.Decomposition(qt@rv, eig, np.asfortranarray(np.linalg.inv(rv))@rt)

    gf_loc_inv = dec.reconstruct(1./hilbert_trafo(dec.xi))
    gf_ii_avg_inv = (diagonal(gf_loc_inv) + diagonal(self_beb_z) - e_onsite) / concentration

    return gf_loc_inv - gf_ii_avg_inv[..., newaxis]*eye


def restrict_self_root_eq(self_beb_z, *args, **kwds):
    """Wrap `self_root_eq` to restrict the solutions to `diagonal(self_cpa_z).imag > 0`."""
    diag_idx = (..., np.eye(*self_beb_z.shape[-2:], dtype=bool))
    self_diag = self_beb_z[diag_idx]
    unphysical = self_diag.imag > 0
    if np.all(~unphysical):  # no need for restrictions
        return self_root_eq(self_beb_z, *args, **kwds)
    distance = self_diag.imag[unphysical].copy()  # distance to physical solution
    self_diag.imag[unphysical] = 0
    self_beb_z.imag[diag_idx] = self_diag.imag
    root = self_root_eq(self_beb_z, *args, **kwds)
    root_diag = root[diag_idx].copy()
    root_diag[unphysical] *= (1 + distance)  # linearly enlarge residues
    # kill unphysical roots
    root_diag.real[unphysical] += 1e-3 * distance * np.where(root_diag.real[unphysical] >= 0, 1, -1)
    root_diag.imag[unphysical] += 1e-3 * distance * np.where(root_diag.imag[unphysical] >= 0, 1, -1)
    root[diag_idx] = root_diag
    return root


def solve_root(z, e_onsite, concentration, hopping, hilbert_trafo: Callable[[complex], complex],
               self_beb_z0=None, restricted=True, **root_kwds):
    """Determine the BEB self-energy by solving the root problem.

    The current implementation doesn't work for rank-deficient problems,
    e.g. `hopping=np.ones(N_cmpt)`. Therefor you should only consider problems
    where `hopping` is sufficiently well conditioned.
    Note that the result should be checked, whether the obtained solution is
    physical.

    Parameters
    ----------
    z : (N_z) complex array_like
        Frequency points.
    e_onsite : (N_cmpt) float or complex np.ndarray
        On-site energy of the components.
    concentration : (N_cmpt) float array_like
        Concentration of the different components.
    hopping : (N_cmpt, N_cmpt) float array_like
        Hopping matrix in the components.
    hilbert_trafo : Callable[[complex], complex]
        Hilbert transformation of the lattice to calculate the local Green's function.
    self_beb_z0 : (..., N_z, N_cmpt, N_cmpt) complex np.ndarray, optional
        Starting guess for the BEB self-energy.
    restricted : bool, optional
        Whether `self_cpa_z` is restricted to `self_cpa_z.imag <= 0`. (default: True)
        Note, that even if `restricted=True`, the imaginary part can get negative
        within tolerance. This should be removed by hand if necessary.
    root_kwds
        Additional arguments passed to `optimize.root`.
        `method` can be used to choose a solver. `options=dict(fatol=tol)` can
        be specified to set the desired tolerance `tol`.

    Returns
    -------
    self_beb_z : (..., N_z, N_cmpt, N_cmpt) complex np.ndarray
        The BEB self-energy as the root of `self_root_eq`

    Raises
    ------
    RuntimeError
        If the root problem cannot be solved.

    Examples
    --------
    >>> from functools import partial
    >>> eps = np.array([-0.5, 0.5])
    >>> c = [0.3, 0.7]
    >>> t = np.array([[1.0, 0.3],
    ...               [0.3, 1.2]])
    >>> hilbert = partial(gt.bethe_hilbert_transform, half_bandwidth=1)

    >>> ww = np.linspace(-1.6, 1.6, num=1000) + 1e-4j
    >>> self_beb_ww = gt.beb.solve_root(ww, e_onsite=eps, concentration=c, hopping=t,
    ...                                 hilbert_trafo=hilbert)
    >>> gf_loc_ww = gt.beb.gf_loc_z(ww, self_beb_ww, hopping=t, hilbert_trafo=hilbert)

    >>> import matplotlib.pyplot as plt
    >>> __ = plt.plot(ww.real, -1./np.pi/c[0]*gf_loc_ww[:, 0].imag, label='A')
    >>> __ = plt.plot(ww.real, -1./np.pi/c[1]*gf_loc_ww[:, 1].imag, label='B')
    >>> __ = plt.plot(ww.real, -1./np.pi*np.sum(gf_loc_ww.imag, axis=-1), label='avg')
    >>> __ = plt.legend()
    >>> plt.show()

    """
    if self_beb_z0 is None:
        self_beb_z0 = np.zeros(z.shape + hopping.shape, dtype=complex)
        # experience shows that a single fixed_point is a good starting point
        self_beb_z0 += self_root_eq(self_beb_z0, z, e_onsite, concentration, hopping, hilbert_trafo)
        if np.all(z.imag >= 0):  # make sure that we are in the retarded regime
            diag_idx = (..., np.eye(*hopping.shape, dtype=bool))
            self_beb_z0[diag_idx] = np.where(self_beb_z0[diag_idx].imag < 0,
                                             self_beb_z0[diag_idx], self_beb_z0[diag_idx].conj())
            assert np.all(self_beb_z0[diag_idx].imag <= 0)
    root_eq = partial(restrict_self_root_eq if restricted else self_root_eq,
                      z=z, e_onsite=e_onsite, concentration=concentration, hopping=hopping,
                      hilbert_trafo=hilbert_trafo)

    method = root_kwds.pop('method', 'krylov')
    if 'callback' not in root_kwds:  # setup LOGGER if no 'callback' is provided
        LOGGER.debug('Search BEB self-energy root')
        root_kwds['callback'] = lambda x, f: LOGGER.debug('Residue: %s', np.linalg.norm(f))

    sol = optimize.root(root_eq, x0=self_beb_z0, method=method, **root_kwds)
    LOGGER.debug("BEB self-energy root found after %s iterations.", sol.nit)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.x
