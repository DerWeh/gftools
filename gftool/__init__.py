# encoding: utf-8
"""Collection of commonly used Green's functions and utilities.

Main purpose is to have a tested base.

Submodules
----------

.. toctree::
   :maxdepth: 1

   gftool.fourier
   gftool.lattice
   gftool.matrix
   gftool.pade

Glossary
--------

.. glossary::

   DOS
      Density of States

   eps
   epsilon
   ϵ
      (Real) energy variable. Typically used for for the :term:`DOS`
      where it replaces the k-dependent Dispersion :math:`ϵ_k`.

   iv
   iν_n
      Bosonic Matsubara frequncies

   iw
   iω_n
      Fermionic Matusbara frequncies

   tau
   τ
      Imaginary time points

   z
      Complex frequency variable

"""
import logging
import warnings

from typing import Callable
from functools import partial
from collections import namedtuple

import numpy as np
import scipy as sp
import scipy.linalg
import scipy.optimize

from numpy import newaxis
from scipy.special import expit, logit

from . import lattice, matrix as gtmatrix
from .basis import pole
from ._version import get_versions

__version__ = get_versions()['version']

LOGGER = logging.getLogger(__name__)

# Bethe lattice
# pylint: disable=wrong-import-position
from .lattice.bethe import (dos as bethe_dos,
                            dos_moment as bethe_dos_moment,
                            gf_d1_z as bethe_gf_d1_z,
                            gf_d2_z as bethe_gf_d2_z,
                            gf_z as bethe_gf_z,
                            hilbert_transform as bethe_hilbert_transform)

# silence warnings of unused imports
assert (bethe_dos and bethe_dos_moment and bethe_gf_d1_z and bethe_gf_d2_z
        and bethe_gf_z and bethe_hilbert_transform)

bethe_dos.m1 = partial(lattice.bethe.dos_moment, 1)
bethe_dos.m2 = partial(lattice.bethe.dos_moment, 2)
bethe_dos.m3 = partial(lattice.bethe.dos_moment, 3)
bethe_dos.m4 = partial(lattice.bethe.dos_moment, 4)
bethe_dos.m5 = partial(lattice.bethe.dos_moment, 5)


# One-dimensional lattice
from .lattice.onedim import (dos as onedim_dos,
                             dos_moment as onedim_dos_moment,
                             gf_z as onedim_gf_z,
                             hilbert_transform as onedim_hilbert_transform)

# silence warnings of unused imports
assert (onedim_dos and onedim_dos_moment and onedim_gf_z and onedim_hilbert_transform)

onedim_dos.m1 = partial(lattice.onedim.dos_moment, 1)
onedim_dos.m2 = partial(lattice.onedim.dos_moment, 2)
onedim_dos.m3 = partial(lattice.onedim.dos_moment, 3)
onedim_dos.m4 = partial(lattice.onedim.dos_moment, 4)
onedim_dos.m5 = partial(lattice.onedim.dos_moment, 5)


# Square lattice
from .lattice.square import (dos as square_dos,
                             dos_moment as square_dos_moment,
                             gf_z as square_gf_z,
                             hilbert_transform as square_hilbert_transform)

# silence warnings of unused imports
assert (square_dos and square_dos_moment and square_gf_z and square_hilbert_transform)

square_dos.m1 = partial(lattice.square.dos_moment, 1)
square_dos.m2 = partial(lattice.square.dos_moment, 2)
square_dos.m3 = partial(lattice.square.dos_moment, 3)
square_dos.m4 = partial(lattice.square.dos_moment, 4)
square_dos.m5 = partial(lattice.square.dos_moment, 5)


def bose_fct(eps, beta):
    r"""Return the Bose function `1/(exp(βϵ)-1)`.

    Parameters
    ----------
    eps : complex or float or array_like
        The energy at which the Bose function is evaluated.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.

    Returns
    -------
    bose_fct : complex of float np.ndarray
        The Bose function, same type as eps.

    """
    betaeps = np.asanyarray(beta*eps)
    res = np.empty_like(betaeps)
    small = betaeps < 700
    res[small] = 1./np.expm1(betaeps[small])
    # avoid overflows for big numbers using negative exponents
    res[~small] = -np.exp(-betaeps[~small])/np.expm1(-betaeps[~small])
    return res


def fermi_fct(eps, beta):
    r"""Return the Fermi function `1/(exp(βϵ)+1)`.

    For complex inputs the function is not as accurate as for real inputs.

    Parameters
    ----------
    eps : complex or float or ndarray
        The energy at which the Fermi function is evaluated.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.

    Returns
    -------
    fermi_fct : complex of float or ndarray
        The Fermi function, same type as eps.

    See Also
    --------
    fermi_fct_inv : The inverse of the Fermi function for real arguments

    """
    z = eps*beta
    try:
        return expit(-z)  # = 0.5 * (1. + tanh(-0.5 * beta * eps))
    except TypeError:
        pass  # complex arguments not handled by expit
    z = np.asanyarray(z)
    pos = z.real > 0
    res = np.empty_like(z)
    res[~pos] = 1./(np.exp(z[~pos]) + 1)
    exp_m = np.exp(-z[pos])
    res[pos] = exp_m/(1 + exp_m)
    return res


def fermi_fct_d1(eps, beta):
    r"""Return the 1st derivative of the Fermi function.

    .. math:: -β\exp(βϵ)/{(\exp(βϵ)+1)}^2

    Parameters
    ----------
    eps : float or float ndarray
        The energy at which the Fermi function is evaluated.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.

    Returns
    -------
    fermi_fct_d1 : float or float ndarray
        The Fermi function, same type as eps.

    See Also
    --------
    fermi_fct

    """
    fermi = fermi_fct(eps, beta=beta)
    return -beta*fermi*(1-fermi)


def fermi_fct_inv(fermi, beta):
    """Inverse of the Fermi function.

    This is e.g. useful for integrals over the derivative of the Fermi function.

    Parameters
    ----------
    fermi : float or float ndarray
        The values of the Fermi function
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.

    Returns
    -------
    fermi_fct_inv : float or float ndarray
        The inverse of the Fermi function `fermi_fct(fermi_fct_inv, beta)=fermi`.

    See Also
    --------
    fermi_fct

    """
    return -logit(fermi)/beta


def matsubara_frequencies(n_points, beta):
    r"""Return *fermionic* Matsubara frequencies :math:`iω_n` for the points `n_points`.

    Parameters
    ----------
    n_points : int array_like
        Points for which the Matsubara frequencies :math:`iω_n` are returned.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.

    Returns
    -------
    matsubara_frequencies : complex ndarray
        Array of the imaginary Matsubara frequencies

    Examples
    --------
    >>> gt.matsubara_frequencies(range(1024), beta=1)
    array([0.+3.14159265e+00j, 0.+9.42477796e+00j, 0.+1.57079633e+01j, ...,
           0.+6.41827379e+03j, 0.+6.42455698e+03j, 0.+6.43084016e+03j])

    """
    n_points = np.asanyarray(n_points).astype(dtype=int, casting='safe')
    return 1j * np.pi / beta * (2*n_points + 1)


def matsubara_frequencies_b(n_points, beta):
    r"""Return *bosonic* Matsubara frequencies :math:`iν_n` for the points `n_points`.

    Parameters
    ----------
    n_points : int ndarray
        Points for which the Matsubara frequencies :math:`iν_n` are returned.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.

    Returns
    -------
    matsubara_frequencies : complex ndarray
        Array of the imaginary Matsubara frequencies

    Examples
    --------
    >>> gt.matsubara_frequencies_b(range(1024), beta=1)
    array([0.+0.00000000e+00j, 0.+6.28318531e+00j, 0.+1.25663706e+01j, ...,
           0.+6.41513220e+03j, 0.+6.42141538e+03j, 0.+6.42769857e+03j])

    """
    n_points = np.asanyarray(n_points).astype(dtype=int, casting='safe')
    return 2j * np.pi / beta * n_points


def pade_frequencies(num: int, beta):
    """Return `num` *fermionic* Padé frequencies :math:`iz_p`.

    The Padé frequencies are the poles of the approximation of the Fermi
    function with `2*num` poles [ozaki2007]_.
    This gives an non-equidistant mesh on the imaginary axis.

    Parameters
    ----------
    num : int
        Number of positive Padé frequencies.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.

    Returns
    -------
    izp : (num) complex np.ndarray
        Positive Padé frequencies.
    resids : (num) float np.ndarray
        Residue of the Fermi function corresponding to `izp`. The residue is
        given relative to true residue of the Fermi function corresponding to
        the poles at Matsubara frequencies. This allows to use Padé frequencies
        as drop-in replacement.
        The actual residues would be `-resids/beta`.

    References
    ----------
    .. [ozaki2007] Ozaki, Taisuke. Continued Fraction Representation of the
       Fermi-Dirac Function for Large-Scale Electronic Structure Calculations.
       Physical Review B 75, no. 3 (January 23, 2007): 035123.
       https://doi.org/10.1103/PhysRevB.75.035123.

    .. [hu2010] J. Hu, R.-X. Xu, and Y. Yan, “Communication: Padé spectrum
       decomposition of Fermi function and Bose function,” J. Chem. Phys., vol.
       133, no. 10, p.  101106, Sep. 2010, https://doi.org/10.1063/1.3484491

    """
    num = 2*num
    a = -np.diagflat(range(1, 2*num, 2))
    b = np.zeros_like(a, dtype=np.float_)
    np.fill_diagonal(b[1:, :], 0.5)
    np.fill_diagonal(b[:, 1:], 0.5)
    eig, v = sp.linalg.eig(a, b=b, overwrite_a=True, overwrite_b=True)
    sort = np.argsort(eig)
    izp = 1j/beta * eig[sort]
    resids = (0.25*v[0]*np.linalg.inv(v)[:, 0]*eig**2)[sort]
    assert np.allclose(-izp[:num//2][::-1], izp[num//2:])
    assert np.allclose(resids[:num//2][::-1], resids[num//2:])
    assert np.all(~np.iscomplex(resids))
    return izp[num//2:], resids.real[num//2:]


def surface_gf_zeps(z, eps, hopping_nn):
    r"""Surface Green's function for stacked layers.

    .. math::
        \left(1 - \sqrt{1 - 4 t^2 g_{00}^2}\right)/(2 t^2 g_{00})

    with :math:`g_{00} = (z-ϵ)^{-1}` [odashima2016]_. This is in principle the
    Green's function for a semi-infinite chain.

    Parameters
    ----------
    z : complex
        Green's function is evaluated at complex frequency `z`.
    eps : float
        Eigenenergy (dispersion) for which the Green's function is evaluated.
    hopping_nn : float
        Nearest neighbor hopping :math:`t` between neighboring layers.

    Returns
    -------
    surface_gf_zeps : complex
        Value of the surface Green's function

    References
    ----------
    .. [odashima2016] Odashima, Mariana M., Beatriz G. Prado, and E. Vernek. Pedagogical
       Introduction to Equilibrium Green's Functions: Condensed-Matter Examples
       with Numerical Implementations. Revista Brasileira de Ensino de Fisica 39,
       no. 1 (September 22, 2016).
       https://doi.org/10.1590/1806-9126-rbef-2016-0087.

    """
    return bethe_gf_z(z-eps, half_bandwidth=2.*hopping_nn)


def hubbard_dimer_gf_z(z, hopping, interaction, kind='+'):
    r"""Green's function for the two site Hubbard model on a *dimer*.

    The Hamilton is given

    .. math:: H = -t∑_{σ}(c^†_{1σ} c_{2σ} + c^†_{2σ} c_{1σ}) + U∑_i n_{i↑} n_{i↓}

    with the `hopping` :math:`t` and the `interaction` :math:`U`.
    The Green's function is given for the operators :math:`c_{±σ} = 1/√2 (c_{1σ} ± c_{2σ})`,
    where :math:`±` is given by `kind`

    Parameters
    ----------
    z : complex ndarray or complex
        Green's function is evaluated at complex frequency `z`
    hopping : float
        The hopping parameter between the sites of the dimer.
    interaction : float
        The Hubbard interaction strength for the on-site interaction.
    kind : {'+', '-'}
        The operator  for which the Green's function is calculated.

    Returns
    -------
    gf_z : complex ndarray
        Value of the Hubbard dimer Green's function at frequencies `z`.

    Notes
    -----
    The solution is obtained by exact digitalization and shown in [eder2017]_.

    References
    ----------
    .. [eder2017] Eder, Robert. “Introduction to the Hubbard Mode.” In The
       Physics of Correlated Insulators, Metals and Superconductors, edited by
       Eva Pavarini, Erik Koch, Richard Scalettar, and Richard Martin.
       Schriften Des Forschungszentrums Jülich Reihe Modeling and Simulation 7.
       Jülich: Forschungszentrum Jülich, 2017.
       https://www.cond-mat.de/events/correl17/manuscripts/eder.pdf.

    """
    if kind not in ('+', '-'):
        raise ValueError("invalid literal for `kind`: '{}'".format(kind))
    s = 1 if kind == '+' else -1
    t = hopping
    U = interaction
    W = (0.25*U*U + 4*t*t)**0.5
    E_0 = 0.5*U - W
    gf_z  = (0.5 + s*t/W) / (z - (E_0 + s*t))
    gf_z += (0.5 - s*t/W) / (z - (U + s*t - E_0))
    return gf_z


# FIXME: write tests for moments
def hubbard_I_self_z(z, U, occ):
    r"""Self-energy in Hubbard-I approximation (atomic solution).

    The chemical potential and the onsite energy have to be included in `z`.

    Parameters
    ----------
    z : complex array_like
        The complex frequencies at which the self-energy is evaluated. `z`
        should be shifted by the onsite energy and the chemical potential.
    U : float array_like
        The local Hubbard interaction `U`.
    occ : float array_like
        The occupation of the opposite spin as the spin of the self-energy.

    Returns
    -------
    Σ_{Hub I} : complex array_like
        The self-energy in Hubbard I approximation.

    Examples
    --------
    >>> U = 5
    >>> mu = U/2  # particle-hole symmetric case -> n=0.5
    >>> ww = np.linspace(-5, 5, num=1000) + 1e-6j

    >>> self_ww = gt.hubbard_I_self_z(ww+mu, U, occ=0.5)

    Show the spectral function for the Bethe lattice,
    we see the two Hubbard bands centered at ±U/2:

    >>> import matplotlib.pyplot as plt
    >>> gf_iw = gt.bethe_gf_z(ww+mu-self_ww, half_bandwidth=1.)
    >>> __ = plt.plot(ww.real, -1./np.pi*gf_iw.imag)
    >>> plt.show()

    """
    hartree = U * occ
    return hartree * z / (z - U + hartree)


pole_gf_z = pole.gf_z
pole_gf_d1_z = pole.gf_d1_z
pole_gf_tau = pole.gf_tau


def pole_gf_tau_b(tau, poles, weights, beta):
    """Bosonic imaginary time Green's function given by a finite number of `poles`.

    The bosonic Green's function is given by
    `G(tau) = -(1 + bose_fct(poles, beta))*exp(-poles*tau)`

    Parameters
    ----------
    tau : (...) float array_like
        Green's function is evaluated at imaginary times `tau`.
        Only implemented for :math:`τ ∈ [0, β]`.
    poles, weights : (..., N) float array_like
        Position and weight of the poles. The real part of the poles needs to
        be positive `poles.real > 0`.
    beta : float
        Inverse temperature

    Returns
    -------
    pole_gf_tau_b : (...) float np.ndarray
        Imaginary time Green's function.

    See Also
    --------
    pole_gf_z : corresponding commutator Green's function

    Raises
    ------
    ValueError
        If any `poles.real <= 0`.

    Examples
    --------
    >>> beta = 10
    >>> tau = np.linspace(0, beta, num=1000)
    >>> gf_tau = gt.pole_gf_tau_b(tau, .5, 1., beta=beta)

    The integrated imaginary time Green's function gives `-np.sum(weights/poles)`

    >>> np.trapz(gf_tau, x=tau)
    -2.0000041750107735

    >>> import matplotlib.pyplot as plt
    >>> __ = plt.plot(tau, gf_tau)
    >>> __ = plt.xlabel('τ')
    >>> plt.show()

    """
    assert np.all((tau >= 0.) & (tau <= beta))
    poles = np.atleast_1d(poles)
    tau = np.asanyarray(tau)[..., newaxis]
    beta = np.asanyarray(beta)[..., newaxis]
    if np.any(poles.real < 0):
        raise ValueError("Bosonic Green's function only well-defined for positive `poles`.")
    # eps((beta-tau)*pole)*g(pole, beta) = -exp(-tau*pole)*g(pole, -beta)
    return np.sum(weights*bose_fct(poles, -beta)*np.exp(-tau*poles), axis=-1)


pole_gf_moments = pole.moments


def chemical_potential(occ_root: Callable[[float], float], mu0=0.0, step0=1.0, **kwds) -> float:
    """Search chemical potential for a given occupation.

    Parameters
    ----------
    occ_root : callable
        Function `occ_root(mu_i) -> occ_i - occ`, which returns the difference
        in occupation for a chemical potential `mu_i` to the given occupation
        `occ`. The sign is important, that `occ_i - occ` is returned.
        Note that the sign is important!
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

    Raises
    ------
    RuntimeError
        If either no bracket can be found (this should only happen for the
        complete empty or completely filled case),
        or if the scalar root search in the bracket fails.

    Notes
    -----
    The search for a chemical potential is here a two-step procedure:
    *First*, we search for a bracket `[mua, mub]` with
    `occ_root(mua) < 0 < occ_root(mub)`. Here we use that the occupation is a
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
    root_res = sp.optimize.root_scalar(occ_root, bracket=bracket, **kwds)
    if not root_res.converged:
        runtime_err = RuntimeError(
            f"Root-search for chemical potential failed after {root_res.iterations}.\n"
            f"Cause of failure: {root_res.flag}"
        )
        runtime_err.mu = root_res.root
        raise runtime_err
    LOGGER.debug("Root found after %s additional evaluations.", root_res.function_calls)
    return root_res.root


Result = namedtuple('Result', ['x', 'err'])


def density(gf_iw, potential, beta, return_err=True, matrix=False, total=False):
    r"""Calculate the number density of the Green's function `gf_iw` at finite temperature `beta`.

    As Green's functions decay only as :math:`1/ω`, the known part of the form
    :math:`1/(iω_n + μ - ϵ - ℜΣ_{\text{static}})` will be calculated analytically.
    :math:`Σ_{\text{static}}` is the ω-independent mean-field part of the self-energy.

    Parameters
    ----------
    gf_iw : complex ndarray
        The Matsubara frequency Green's function for positive frequencies :math:`iω_n`.
        The last axis corresponds to the Matsubara frequencies.
    potential : float ndarray or float
        The static potential for the large-ω behavior of the Green's function.
        It is the real constant :math:`μ - ϵ - ℜΣ_{\text{static}}`.
        The shape must agree with `gf_iw` without the last axis.
        If `matrix`, then potential needs to be a (N, N) matrix. It is the
        negative of the Hamiltonian matrix and thus needs to be hermitian.
    beta : float
        The inverse temperature `beta` = 1/T.
    return_err : bool or float, optional
        If `True` (default), the error estimate will be returned along with the density.
        If `return_err` is a float, a warning will Warning will be issued if
        the error estimate is larger than `return_err`. If `False`, no error
        estimate is calculated.
        See `density_error` for description of the error estimate.
    total : bool or tuple
        If `total` the total density (summed over all dimensions) is returned.
        Also a tuple can be given, over which axes the sums is taken.

    Returns
    -------
    x : float
        The number density of the given Green's function `gf_iw`.
    err : float
        An estimate for the density error. Only returned if `return_err` is `True`.

    Notes
    -----
    The number density can be obtained from the Matsubara frequency Green's function using

    .. math:: ⟨n⟩ = \lim_{ϵ↗0} G(τ=-ϵ) = 1/β ∑_{n=-∞}^{∞} G(iω_n)

    As Green's functions decay only as :math:`O(1/ω)`, truncation of the summation
    yields a non-vanishing contribution of the tail.
    For the analytic structure of the Green's function see [2]_, [3]_.
    To take this into consideration the known part of the form
    :math:`1/(iω_n + μ - ϵ - ℜΣ_{\text{static}})` will be calculated analytically.
    This yields [1]_

    .. math::

       ⟨n⟩ = 1/β ∑_{n=-∞}^{∞} [G(iω_n) - 1/(iω_n + μ - ϵ - ℜΣ_{\text{static}})] \\
             + 1/2 + 1/2 \tanh[1/2 β(μ - ϵ - ℜΣ_{\text{static}})].

    We can use the symmetry :math:`G(z*) = G^*(z)` do reduce the sum only over
    positive Matsubara frequencies

    .. math::

       ∑_{n=-∞}^{∞} G(iω_n)
          &= ∑_{n=-∞}^{-1} G(iω_n) + ∑_{n=0}^{n=∞} G(iω_n) \\
          &= ∑_{n=0}^{∞} [G(-iω_n) + G(iω_n)] \\
          &= 2 ∑_{n=0}^{∞} ℜG(iω_n).

    Thus we get the final expression

    .. math::
       ⟨n⟩ = 2/β ∑_{n≥0} ℜ[G(iω_n) - 1/(iω_n + μ - ϵ - ℜΣ_{\text{static}})] \\
             + 1/2 + 1/2 \tanh[1/2 β(μ - ϵ - ℜΣ_{\text{static}})].

    References
    ----------
    .. [1] Hale, S. T. F., and J. K. Freericks. "Many-Body Effects on the
       Capacitance of Multilayers Made from Strongly Correlated Materials."
       Physical Review B 85, no. 20 (May 24, 2012).
       https://doi.org/10.1103/PhysRevB.85.205444.
    .. [2] Eder, Robert. “Introduction to the Hubbard Mode.” In The Physics of
       Correlated Insulators, Metals and Superconductors, edited by Eva
       Pavarini, Erik Koch, Richard Scalettar, and Richard Martin. Schriften
       Des Forschungszentrums Jülich Reihe Modeling and Simulation 7. Jülich:
       Forschungszentrum Jülich, 2017.
       https://www.cond-mat.de/events/correl17/manuscripts/eder.pdf.
    .. [3] Luttinger, J. M. “Analytic Properties of Single-Particle Propagators
       for Many-Fermion Systems.” Physical Review 121, no. 4 (February 15,
       1961): 942–49. https://doi.org/10.1103/PhysRev.121.942.

    """
    iw = matsubara_frequencies(np.arange(gf_iw.shape[-1]), beta=beta)
    if total:
        assert gf_iw.ndim == 1

    if matrix:
        dec = gtmatrix.decompose_hamiltonian(potential)
        xi = dec.xi
        tail = dec.reconstruct(1./np.add.outer(iw, xi), kind='diag')
        tail = np.moveaxis(tail, source=0, destination=-1)
        analytic = dec.reconstruct(fermi_fct(-xi, beta=beta), kind='diag')
    else:
        tail = 1/np.add.outer(potential, iw)
        analytic = fermi_fct(-potential, beta=beta)

    if total:
        axis = tuple(range(tail.ndim - 1)) if total is True else total
        tail = tail.real.sum(axis=axis)
        analytic = np.sum(analytic)

    delta_g_re = gf_iw.real - tail.real
    density = 2. * np.sum(delta_g_re, axis=-1) / beta
    density += analytic
    if return_err:
        err = density_error(delta_g_re, iw)
        if return_err is True:
            return Result(x=density, err=err)
        else:
            if np.any(err > return_err):
                warnings.warn("density result inaccurate, error estimate = "
                              + str(err), Warning)
    return density


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
        `G(z) = moments / z**np.arange(N)` for large `z`.
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
        mom_gf = pole.PoleGf.from_z(iws, gf_iw[..., newaxis, :], n_pole=n_fit+n_mom,
                                    moments=moments, width=None, weight=weight)
    else:
        mom_gf = pole.PoleGf.from_moments(moments, width=None)
    delta_gf_iw = gf_iw.real - mom_gf.eval_z(iws).real
    return 2./beta*np.sum(weights * delta_gf_iw.real, axis=-1) + mom_gf.occ(beta)[..., 0]


def density_error(delta_gf_iw, iw_n, noisy=True):
    """Return an estimate for the upper bound of the error in the density.

    This estimate is based on the *integral test*. The crucial assumption is,
    that `ω_N` is large enough, such that :math:`ΔG ∼ 1/ω_n^2` for all larger
    :math:`n`.
    If this criteria is not met, the error estimate is unreasonable and can
    **not** be trusted. If the error is of the same magnitude as the density
    itself, the behavior of the variable `factor` should be checked.

    Parameters
    ----------
    delta_gf_iw : (..., N) ndarray
        The difference between the Green's function :math:`Δ G(iω_n)`
        and the non-interacting high-frequency estimate. Only it's real part is
        needed.
    iw_n : (N) complex ndarray
        The Matsubara frequencies corresponding to `delta_gf_iw`.

    Returns
    -------
    estimate : float
        The estimate of the upper bound of the error. Reliable only for large
        enough Matsubara frequencies.

    """
    part = slice(iw_n.size//10, None, None)  # only consider last 10, iw must be big
    wn = iw_n[part].imag
    denominator = 1./np.pi/wn[-1]
    if noisy:
        factor = np.average(delta_gf_iw[..., part] * wn**2, axis=-1)
    else:
        delta_gf_iw = abs(delta_gf_iw.real)
        factor = np.max(delta_gf_iw[..., part] * wn**2, axis=-1)
    estimate = factor * denominator
    return estimate


def density_error2(delta_gf_iw, iw_n):
    """Return an estimate for the upper bound of the error in the density.

    This estimate is based on the *integral test*. The crucial assumption is,
    that `ω_N` is large enough, such that :math:`ΔG ∼ 1/ω_n^3` for all larger
    :math:`n`.
    If this criteria is not met, the error estimate is unreasonable and can
    **not** be trusted. If the error is of the same magnitude as the density
    itself, the behavior of the variable `factor` should be checked.

    Parameters
    ----------
    delta_gf_iw : (..., N) ndarray
        The difference between the Green's function :math:`Δ G(iω_n)`
        and the non-interacting high-frequency estimate. Only it's real part is
        needed.
    iw_n : (N) complex ndarray
        The Matsubara frequencies corresponding to `delta_gf_iw`.

    Returns
    -------
    estimate : float
        The estimate of the upper bound of the error. Reliable only for large
        enough Matsubara frequencies.

    """
    delta_gf_iw = abs(delta_gf_iw.real)
    part = slice(iw_n.size//10, None, None)  # only consider last 10, iw must be big
    wn = iw_n[part].imag
    denominator = 1./2.*np.pi/wn[-1]**2
    factor = np.max(delta_gf_iw[..., part] * wn**3, axis=-1)
    estimate = factor * denominator
    return estimate


def check_convergence(gf_iw, potential, beta, order=2, matrix=False, total=False):
    """Return data for visual inspection of  the density error.

    The calculation of the density error assumes that *sufficient* Matsubara
    frequencies were used. Sufficient means here, that the reminder :math:`ΔG`
    does **not** grow anymore. If the error estimate is small, but
    `check_convergence` returns rapidly growing data, the number of Matsubara
    frequencies is not sufficient

    Parameters
    ----------
    see `density`

    Other Parameters
    ----------------
    order : int
        The assumed order of the first non-vanishing term of the Laurent expansion.

    Returns
    -------
    check_convergence : float ndarray
        The last dimension of `check_convergence` corresponds to the Matsubara
        frequencies.

    """
    iw = matsubara_frequencies(np.arange(gf_iw.shape[-1]), beta=beta)

    if matrix:
        dec = gtmatrix.decompose_hamiltonian(potential)
        tail = dec.reconstruct(1./np.add.outer(dec.xi, iw), kind='diag')
    else:
        tail = 1/np.add.outer(potential, iw)

    if total:
        tail = tail.real.sum(axis=tuple(range(tail.ndim - 1)))

    delta_g_re = gf_iw.real - tail.real
    return iw.imag**order * delta_g_re
