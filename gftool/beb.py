"""
Blackman, Esterling, and Berk (BEB) approach to off-diagonal disorder.

It extends CPA allowing for random hopping amplitudes. [blackman1971]_

The implementation is based on a SVD of the `hopping` matrix,
which is the dimensionless scaling of the hopping of the components. [weh2021]_
However, we use the unitary eigendecomposition instead of the SVD.

Physical quantities
-------------------
The main quantity of interest is the average local Green's function `gf`.

First the effective medium `self_beb_z` has to be calculated using `solve_root`.
With this result the Green's function can be calculated by the function `gf_loc_z`.

In the BEB formalism, the local Green's function `gf` is a matrix in the components.
The self-consistent Green's function `gf` is diagonal, its trace is the average
physical Green's function.
If only the non-vanishing diagonal elements have been calculated `gf=gf_loc_z(..., diag=True)`,
the average Green's function is `np.sum(gf, axis=-1)`.
The diagonal elements of `gf` are the average for a specific component
(conditional average) multiplied by the concentration of that component.

References
----------
.. [blackman1971]
   Blackman, J.A., Esterling, D.M., Berk, N.F., 1971.
   Generalized Locator---Coherent-Potential Approach to Binary Alloys.
   Phys. Rev. B 4, 2412-2428. https://doi.org/10.1103/PhysRevB.4.2412
.. [weh2021] Weh, A., Zhang, Y., Östlin, A., Terletska, H., Bauernfeind, D.,
   Tam, K.-M., Evertz, H.G., Byczuk, K., Vollhardt, D., Chioncel, L., 2021.
   Dynamical mean-field theory of the Anderson--Hubbard model with local and
   nonlocal disorder in tensor formulation. Phys. Rev. B 104, 045127.
   https://doi.org/10.1103/PhysRevB.104.045127

Examples
--------
We consider a Bethe lattice with two components 'A' and 'B'.
The have the on-site energies `-0.5` and `0.5` respectively,
the concentrations `0.3` and `0.7`.
Furthermore, we assume that the hopping amplitude between 'A' and 'B' is only
`0.3` times the hopping between two 'A' sites,
while the hopping between two 'B' sites is `1.2` times the hopping between two
'A' sites.

Then the following code calculates the local Green's function for component 'A'
and 'B' (conditionally averaged) as well as the average Green's function of the
system.

.. plot::

    from functools import partial

    import gftool as gt
    import numpy as np
    import matplotlib.pyplot as plt

    eps = np.array([-0.5, 0.5])
    c = np.array([0.3, 0.7])
    t = np.array([[1.0, 0.3],
                  [0.3, 1.2]])
    hilbert = partial(gt.bethe_hilbert_transform, half_bandwidth=1)

    ww = np.linspace(-1.6, 1.6, num=1000) + 1e-4j
    self_beb_ww = gt.beb.solve_root(ww, e_onsite=eps, concentration=c, hopping=t,
                                    hilbert_trafo=hilbert)
    gf_loc_ww = gt.beb.gf_loc_z(ww, self_beb_ww, hopping=t, hilbert_trafo=hilbert)

    __ = plt.plot(ww.real, -1./np.pi/c[0]*gf_loc_ww[:, 0].imag, label='A')
    __ = plt.plot(ww.real, -1./np.pi/c[1]*gf_loc_ww[:, 1].imag, label='B')
    __ = plt.plot(ww.real, -1./np.pi*np.sum(gf_loc_ww.imag, axis=-1), ':', label='avg')
    __ = plt.legend()
    plt.show()
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Callable

import numpy as np
from numpy import newaxis
from scipy import optimize

from gftool.matrix import UDecomposition, decompose_her, decompose_mat

LOGGER = logging.getLogger(__name__)

# gu-function versions to extract diagonal and transpose matrices
diagonal = partial(np.diagonal, axis1=-2, axis2=-1)
transpose = partial(np.swapaxes, axis1=-1, axis2=-2)


class SpecDec(UDecomposition):
    """
    SVD like spectral decomposition.

    Works only for N×N matrices unlike the `UDecomposition` base class.

    Parameters
    ----------
    rv : (..., N, N) complex np.ndarray
        The matrix of right eigenvectors.
    eig : (..., N) float np.ndarray
        The vector of real eigenvalues.
    rv_inv : (..., N, N) complex np.ndarray
        The inverse of `rv`.
    """

    def truncate(self, rcond=None) -> SpecDec:
        """
        Return the truncated spectral decomposition.

        Singular values smaller than `rcond` times the largest singular values
        are discarded.

        Parameters
        ----------
        rcond : float, rcond
            Cut-off ratio for small singular values.

        Returns
        -------
        SpecDec
            The truncates the spectral decomposition discarding small singular values.
        """
        if rcond is None:
            rcond = np.finfo(self.eig.dtype).eps * max(self.u.shape[-2:])
        max_eig = np.max(abs(self.eig), axis=-1)
        significant = abs(self.eig) > max_eig*rcond
        return self.__class__(rv=self.rv[..., :, significant], eig=self.eig[..., significant],
                              rv_inv=self.rv_inv[..., significant, :])

    @property
    def is_trunacted(self) -> bool:
        """Check if SVD of square matrix is truncated/compact or full."""
        ushape, uhshape = self.u.shape, self.uh.shape
        return not ushape[-2] == ushape[-1] == uhshape[-2]

    def partition(self, return_sqrts=False):
        """
        Symmetrically partition the spectral decomposition as `u * eig**0.5, eig**0.5 * uh`.

        If `return_sqrts` then `us, np.sqrt(s), suh` is returned,
        else only `us, suh` is returned (default: False).
        """
        sqrt_eig = np.emath.sqrt(self.eig)
        us, suh = self.u * sqrt_eig[..., newaxis, :], sqrt_eig[..., :, newaxis] * self.uh
        if return_sqrts:
            return us, sqrt_eig, suh
        return us, suh


def gf_loc_z(z, self_beb_z, hopping, hilbert_trafo: Callable[[complex], complex],
             diag=True, rcond=None):
    """
    Calculate average local Green's function matrix in components.

    For the self-consistent self-energy `self_beb_z` it is diagonal in the
    components. Note, that `gf_loc_z` contain the `concentration`.

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
        matrix (default: True).
    rcond : float, optional
        Cut-off ratio for small singular values of `hopping`. For the purposes
        of rank determination, singular values are treated as zero if they are
        smaller than `rcond` times the largest singular value of `hopping`.

    Returns
    -------
    (..., N_cmpt) or (..., N_cmpt, N_cmpt) complex np.ndarray
        The average local Green's function matrix.

    See Also
    --------
    solve_root
    """
    hopping_dec = SpecDec(*decompose_her(hopping))
    LOGGER.info('hopping singular values %s', hopping_dec.s)
    hopping_dec = hopping_dec.truncate(rcond)
    LOGGER.info('Keeping %s (out of %s)', hopping_dec.s.shape[-1], hopping_dec.uh.shape[-1])
    kind = 'diag' if diag else 'full'

    eye = np.eye(*hopping.shape)
    us, sqrt_s, suh = hopping_dec.partition(return_sqrts=True)
    # [..., newaxis]*eye add matrix axis
    z_m_self = z[..., newaxis, newaxis]*eye - self_beb_z
    z_m_self_inv = np.asfortranarray(np.linalg.inv(z_m_self))
    dec = decompose_mat(suh @ z_m_self_inv @ us)
    diag_inv = 1. / dec.eig
    if not hopping_dec.is_trunacted:
        svh_inv = transpose(hopping_dec.uh).conj() / sqrt_s[..., newaxis, :]
        us_inv = transpose(hopping_dec.u).conj() / sqrt_s[..., :, newaxis]
        dec.rv = svh_inv @ np.asfortranarray(dec.rv)
        dec.rv_inv = np.asfortranarray(dec.rv_inv) @ us_inv
        return dec.reconstruct(hilbert_trafo(diag_inv), kind=kind)

    dec.rv = z_m_self_inv @ us @ np.asfortranarray(dec.rv)
    dec.rv_inv = np.asfortranarray(dec.rv_inv) @ suh @ z_m_self_inv
    correction = dec.reconstruct((diag_inv*hilbert_trafo(diag_inv) - 1) * diag_inv, kind=kind)
    return (diagonal(z_m_self_inv) if diag else z_m_self_inv) + correction


def self_root_eq(self_beb_z, z, e_onsite, concentration, hopping_dec: SpecDec,
                 hilbert_trafo: Callable[[complex], complex]):
    """
    Root equation r(Σ)=0 for BEB.

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
    hopping_dec : SVD
        Compact SVD decomposition of the (N_cmpt, N_cmpt) hopping matrix in the
        components.
    hilbert_trafo : Callable[[complex], complex]
        Hilbert transformation of the lattice to calculate the local Green's function.

    Returns
    -------
    (..., N_cmpt, N_cmpt)
        Difference of the inverses of the local and the average Green's function.
        If `diff = 0`, `self_beb_z` is the correct self-energy.

    See Also
    --------
    solve_root
    """
    eye = np.eye(e_onsite.shape[-1])  # [..., newaxis]*eye adds matrix axis
    z_m_self = z[..., newaxis, newaxis]*eye - self_beb_z
    # split symmetrically
    us, suh = hopping_dec.partition()
    # matrix-products are faster if larger arrays are in Fortran order
    z_m_self_inv = np.asfortranarray(np.linalg.inv(z_m_self))
    dec = decompose_mat(suh @ z_m_self_inv @ us)
    dec.rv = us @ np.asfortranarray(dec.rv)
    dec.rv_inv = np.asfortranarray(dec.rv_inv) @ suh
    diag_inv = 1. / dec.eig
    if not hopping_dec.is_trunacted:
        gf_loc_inv = dec.reconstruct(1./hilbert_trafo(diag_inv), kind='full')
    else:
        gf_loc_inv = z_m_self + dec.reconstruct(1./hilbert_trafo(diag_inv) - diag_inv, kind='full')

    gf_ii_avg_inv = (diagonal(gf_loc_inv) + diagonal(self_beb_z) - e_onsite) / concentration

    return gf_loc_inv - gf_ii_avg_inv[..., newaxis]*eye


def restrict_self_root_eq(self_beb_z, *args, **kwds):
    """Wrap `self_root_eq` to restrict the solutions to `diagonal(self_beb_z).imag > 0`."""
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
    """
    Determine the BEB self-energy by solving the root problem.

    Note, that the result should be checked, whether the obtained solution is
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
        Whether the diagonal of `self_beb_z` is restricted to `self_beb_z.imag <= 0`
        (default: True).
        Note, that even if `restricted=True`, the imaginary part can get
        negative within tolerance. This should be removed by hand if necessary.
    rcond : float, optional
        Cut-off ratio for small singular values of `hopping`. For the purposes
        of rank determination, singular values are treated as zero if they are
        smaller than `rcond` times the largest singular value of `hopping`.
    **root_kwds
        Additional arguments passed to `scipy.optimize.root`.
        `method` can be used to choose a solver. `options=dict(fatol=tol)` can
        be specified to set the desired tolerance `tol`.

    Returns
    -------
    (..., N_cmpt, N_cmpt) complex np.ndarray
        The BEB self-energy as the root of `self_root_eq`.

    Raises
    ------
    RuntimeError
        If the root problem cannot be solved.

    See Also
    --------
    gf_loc_z

    Notes
    -----
    The root problem is solved for the complete input simultaneously.
    This provides a speed up as the code is vectorized, however, it comes
    with the trade-off of complicating the root search.
    So in some cases, it makes sense to split the input arrays, and calculate
    the root separately.

    The default method is 'krylov', which typically does a good job.
    In some cases 'excitingmixing' was found to do a better job,
    especially close to the CPA limit, where some singular values become small.

    The progress of the root search is logged for the `logging.DEBUG` level.

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
    hopping_dec = SpecDec(*decompose_her(hopping))
    LOGGER.info('hopping singular values %s', hopping_dec.s)
    hopping_dec = hopping_dec.truncate(rcond)
    LOGGER.info('Keeping %s (out of %s)', hopping_dec.s.shape[-1], hopping_dec.uh.shape[-1])
    self_root_part = partial(self_root_eq, z=z, e_onsite=e_onsite, concentration=concentration,
                             hopping_dec=hopping_dec, hilbert_trafo=hilbert_trafo)
    if self_beb_z0 is None:
        self_beb_z0 = np.zeros(hopping.shape, dtype=complex)
        # experience shows that a single fixed_point is a good starting point
        self_beb_z0 = self_root_part(self_beb_z0)
        if np.all(z.imag >= 0):  # make sure that we are in the retarded regime
            diag_idx = (..., np.eye(*hopping.shape, dtype=bool))
            self_beb_z0[diag_idx] = np.where(self_beb_z0[diag_idx].imag < 0,
                                             self_beb_z0[diag_idx], self_beb_z0[diag_idx].conj())
            assert np.all(self_beb_z0[diag_idx].imag <= 0)
    else:  # to use in root, self_beb_z0 has to have the correct shape
        output = np.broadcast(z, e_onsite[..., 0], concentration[..., 0], self_beb_z0[..., 0, 0])
        self_beb_z0 = np.broadcast_to(self_beb_z0, shape=output.shape + np.asarray(hopping).shape)
    root_eq = partial(restrict_self_root_eq if restricted else self_root_eq,
                      **self_root_part.keywords)

    root_kwds.setdefault("method", "krylov")
    LOGGER.debug('Search BEB self-energy root')
    if 'callback' not in root_kwds and LOGGER.isEnabledFor(logging.DEBUG):
        # setup LOGGER if no 'callback' is provided
        root_kwds['callback'] = lambda _, f: LOGGER.debug('Residue: %s', np.linalg.norm(f))

    sol = optimize.root(root_eq, x0=self_beb_z0, **root_kwds)
    LOGGER.info("BEB self-energy root found after %s iterations.", sol.nit)

    if LOGGER.isEnabledFor(logging.INFO):
        # check condition number in matrix diagonalization to make sure it is well defined
        us, suh = hopping_dec.partition()
        z_m_self = z[..., newaxis, newaxis]*np.eye(*hopping.shape) - sol.x
        dec = decompose_mat(suh @ np.linalg.inv(z_m_self) @ us)
        max_cond = np.max(np.linalg.cond(dec.rv))
        LOGGER.info("Maximal coordination number for diagonalization: %s", max_cond)

    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.x
