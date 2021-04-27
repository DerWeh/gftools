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

from typing import Callable, NamedTuple
from functools import partial

import numpy as np

from scipy import optimize
from numpy import newaxis

from gftool import matrix

LOGGER = logging.getLogger(__name__)

# gu-function versions to extract diagonal and transpose matrices
diagonal = partial(np.diagonal, axis1=-2, axis2=-1)
transpose = partial(np.swapaxes, axis1=-1, axis2=-2)


class SVD(NamedTuple):
    """Container for the singular value decomposition of a matrix."""

    u: np.ndarray
    s: np.ndarray
    vh: np.ndarray

    def truncate(self, rcond=None) -> "SVD":
        """Return the truncated singular values decomposition.

        Singular values smaller than `rcond` times the largest singular values
        are discarded.

        Parameters
        ----------
        rcond : float, rcond
            Cut-off ratio for small singular values.

        Returns
        -------
        truncated_svd : SVD
            The truncates singular value decomposition.

        """
        if rcond is None:
            rcond = np.finfo(self.s.dtype).eps * max(self.u.shape[-2:])
        significant = self.s > self.s[..., 0]*rcond
        return SVD(u=self.u[..., :, significant], s=self.s[..., significant],
                   vh=self.vh[..., significant, :])


def gf_loc_z(z, self_beb_z, hopping, hilbert_trafo: Callable[[complex], complex],
             diag=True, rcond=None):
    """Calculate average local Green's function matrix in components.

    For the self-consistent self-energy `self_beb_z` this it is diagonal in the
    components. Not that `gf_loc_z` implicitly contains the concentrations.

    Parameters
    ----------
    z : (...) complex np.ndarray
        Frequency points.
    self_beb_z : (..., N_cmpt, N_cmpt) complex np.ndarray
        BEB self-energy.
    hopping : (N_cmpt, N_cmpt) float array_like
        Hopping matrix in the components.
    hilbert_trafo : Callable[[complex], complex]
        Hilbert transformation of the lattice to calculate the local Green's function.
    diag : bool, optional
        If `diag`, only the diagonal elements are calculated, else the full
        matrix. (default: True)
    rcond : float, optional
        Cut-off ratio for small singular values of `hopping`. For the purposes
        of rank determination, singular values are treated as zero if they are
        smaller than `rcond` times the largest singular value of `hopping`.

    Returns
    -------
    gf_loc_z : (..., N_cmpt) or (..., N_cmpt, N_cmpt) complex np.ndarray
        The average local Green's function matrix.

    """
    hopping_svd = SVD(*np.linalg.svd(hopping, hermitian=True))
    LOGGER.info('hopping singular values %s', hopping_svd.s)
    hopping_svd = hopping_svd.truncate(rcond)
    LOGGER.info('Keeping %s (out of %s)', hopping_svd.s.shape[-1], hopping_svd.vh.shape[-1])
    kind = 'diag' if diag else 'full'

    eye = np.eye(*hopping.shape)
    sqrt_s = np.sqrt(hopping_svd.s)
    us, svh = hopping_svd.u * sqrt_s[..., newaxis, :], sqrt_s[..., :, newaxis] * hopping_svd.vh
    # [..., newaxis]*eye add matrix axis
    z_m_self = z[..., newaxis, newaxis]*eye - self_beb_z
    z_m_self_inv = np.asfortranarray(np.linalg.inv(z_m_self))
    dec = matrix.Decomposition.from_gf(svh @ z_m_self_inv @ us)
    diag_inv = 1. / dec.xi
    if us.shape[-2] == us.shape[-1]:  # square matrix -> not truncated (full rank)
        svh_inv = transpose(hopping_svd.vh).conj() / sqrt_s[..., newaxis, :]
        us_inv = transpose(hopping_svd.u).conj() / sqrt_s[..., :, newaxis]
        dec.rv = svh_inv @ np.asfortranarray(dec.rv)
        dec.rv_inv = np.asfortranarray(dec.rv_inv) @ us_inv
        return dec.reconstruct(hilbert_trafo(diag_inv), kind=kind)

    dec.rv = z_m_self_inv @ us @ np.asfortranarray(dec.rv)
    dec.rv_inv = np.asfortranarray(dec.rv_inv) @ svh @ z_m_self_inv
    correction = dec.reconstruct((diag_inv*hilbert_trafo(diag_inv) - 1) * diag_inv, kind=kind)
    return (diagonal(z_m_self_inv) if diag else z_m_self_inv) + correction


def self_root_eq(self_beb_z, z, e_onsite, concentration, hopping_svd: SVD,
                 hilbert_trafo: Callable[[complex], complex]):
    """Root equation r(Σ)=0 for BEB.

    Parameters
    ----------
    self_beb_z : (..., N_cmpt, N_cmpt) complex np.ndarray
        BEB self-energy.
    z : (...) complex np.ndarray
        Frequency points.
    e_onsite : (..., N_cmpt) float or complex array_like
        On-site energy of the components.
    concentration : (..., N_cmpt) float array_like
        Concentration of the different components.
    hopping_svd : SVD
        Compact SVD decomposition of the (N_cmpt, N_cmpt) hopping matrix in the
        components.
    hilbert_trafo : Callable[[complex], complex]
        Hilbert transformation of the lattice to calculate the local Green's function.

    Returns
    -------
    diff : (..., N_cmpt, N_cmpt)
        Difference of the inverses of the local and the average Green's function.
        If `diff = 0`, `self_beb_z` is the correct self-energy.

    """
    eye = np.eye(e_onsite.shape[-1])  # [..., newaxis]*eye adds matrix axis
    z_m_self = z[..., newaxis, newaxis]*eye - self_beb_z
    # split symmetrically
    sqrt_s = np.sqrt(hopping_svd.s)
    us, svh = hopping_svd.u * sqrt_s[..., newaxis, :], sqrt_s[..., :, newaxis] * hopping_svd.vh
    # matrix-products are faster if larger arrays are in Fortran order
    z_m_self_inv = np.asfortranarray(np.linalg.inv(z_m_self))
    dec = matrix.Decomposition.from_gf(svh @ z_m_self_inv @ us)
    dec.rv = us @ np.asfortranarray(dec.rv)
    dec.rv_inv = np.asfortranarray(dec.rv_inv) @ svh
    diag_inv = 1. / dec.xi
    if us.shape[-2] == us.shape[-1]:  # square matrix -> not truncated
        gf_loc_inv = dec.reconstruct(1./hilbert_trafo(diag_inv), kind='full')
    else:
        gf_loc_inv = z_m_self + dec.reconstruct(1./hilbert_trafo(diag_inv) - diag_inv, kind='full')

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
               self_beb_z0=None, restricted=True, rcond=None, **root_kwds):
    """Determine the BEB self-energy by solving the root problem.

    Note that the result should be checked, whether the obtained solution is
    physical.

    Parameters
    ----------
    z : (...) complex np.ndarray
        Frequency points.
    e_onsite : (..., N_cmpt) float or complex np.ndarray
        On-site energy of the components.
    concentration : (..., N_cmpt) float np.ndarray
        Concentration of the different components.
    hopping : (N_cmpt, N_cmpt) float array_like
        Hopping matrix in the components.
    hilbert_trafo : Callable[[complex], complex]
        Hilbert transformation of the lattice to calculate the local Green's function.
    self_beb_z0 : (..., N_cmpt, N_cmpt) complex np.ndarray, optional
        Starting guess for the BEB self-energy.
    restricted : bool, optional
        Whether the diagonal of `self_beb_z` is restricted to `self_beb_z.imag <= 0`.
        (default: True)
        Note, that even if `restricted=True`, the imaginary part can get
        negative within tolerance. This should be removed by hand if necessary.
    rcond : float, optional
        Cut-off ratio for small singular values of `hopping`. For the purposes
        of rank determination, singular values are treated as zero if they are
        smaller than `rcond` times the largest singular value of `hopping`.
    root_kwds
        Additional arguments passed to `optimize.root`.
        `method` can be used to choose a solver. `options=dict(fatol=tol)` can
        be specified to set the desired tolerance `tol`.

    Returns
    -------
    self_beb_z : (..., N_cmpt, N_cmpt) complex np.ndarray
        The BEB self-energy as the root of `self_root_eq`

    Raises
    ------
    RuntimeError
        If the root problem cannot be solved.

    Examples
    --------
    >>> from functools import partial
    >>> eps = np.array([-0.5, 0.5])
    >>> c = np.array([0.3, 0.7])
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
    hopping_svd = SVD(*np.linalg.svd(hopping, hermitian=True))
    LOGGER.info('hopping singular values %s', hopping_svd.s)
    hopping_svd = hopping_svd.truncate(rcond)
    LOGGER.info('Keeping %s (out of %s)', hopping_svd.s.shape[-1], hopping_svd.vh.shape[-1])
    if self_beb_z0 is None:
        self_beb_z0 = np.zeros(hopping.shape, dtype=complex)
        # experience shows that a single fixed_point is a good starting point
        self_beb_z0 = self_root_eq(self_beb_z0, z, e_onsite, concentration,
                                   hopping_svd, hilbert_trafo)
        if np.all(z.imag >= 0):  # make sure that we are in the retarded regime
            diag_idx = (..., np.eye(*hopping.shape, dtype=bool))
            self_beb_z0[diag_idx] = np.where(self_beb_z0[diag_idx].imag < 0,
                                             self_beb_z0[diag_idx], self_beb_z0[diag_idx].conj())
            assert np.all(self_beb_z0[diag_idx].imag <= 0)
    else:  # to use in root, self_beb_z0 has to have the correct shape
        # dirty hack: do one iteration to get the correct shape
        self_beb_z0 = (self_beb_z0
                       + 0*self_root_eq(self_beb_z0, z, e_onsite, concentration,
                                        hopping_svd, hilbert_trafo))
    root_eq = partial(restrict_self_root_eq if restricted else self_root_eq,
                      z=z, e_onsite=e_onsite, concentration=concentration,
                      hopping_svd=hopping_svd, hilbert_trafo=hilbert_trafo)

    method = root_kwds.pop('method', 'krylov')
    if 'callback' not in root_kwds:  # setup LOGGER if no 'callback' is provided
        LOGGER.debug('Search BEB self-energy root')
        root_kwds['callback'] = lambda x, f: LOGGER.debug('Residue: %s', np.linalg.norm(f))

    sol = optimize.root(root_eq, x0=self_beb_z0, method=method, **root_kwds)
    LOGGER.debug("BEB self-energy root found after %s iterations.", sol.nit)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.x
