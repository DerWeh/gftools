"""
Collection of commonly used Green's functions and utilities.

Main purpose is to have a tested base.

Submodules
----------

.. autosummary::
  :toctree: generated

   gftool.basis
   gftool.beb
   gftool.cpa
   gftool.fourier
   gftool.hermpade
   gftool.lattice
   gftool.linalg
   gftool.linearprediction
   gftool.matrix
   gftool.pade
   gftool.polepade
   gftool.siam

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
      Bosonic Matsubara frequencies

   iw
   iω_n
      Fermionic Matsubara frequencies

   tau
   τ
      Imaginary time points

   z
      Complex frequency variable
"""

import logging
import warnings
from collections import namedtuple

import numpy as np
from numpy import newaxis

import gftool.siam

from . import beb, cpa, fourier, lattice, linearprediction, polepade
from . import matrix as gtmatrix
from ._util import _gu_sum
from ._version import __version__
from .basis.pole import gf_d1_z as pole_gf_d1_z
from .basis.pole import gf_ret_t as pole_gf_ret_t
from .basis.pole import gf_tau as pole_gf_tau

# Green's function give by finite number poles
from .basis.pole import gf_z as pole_gf_z
from .basis.pole import moments as pole_gf_moments
from .density import chemical_potential, density_iw

# Body-centered cubic lattice
from .lattice.bcc import dos as bcc_dos
from .lattice.bcc import dos_moment as bcc_dos_moment
from .lattice.bcc import gf_z as bcc_gf_z
from .lattice.bcc import hilbert_transform as bcc_hilbert_transform

# Bethe lattice
from .lattice.bethe import dos as bethe_dos
from .lattice.bethe import dos_moment as bethe_dos_moment
from .lattice.bethe import gf_d1_z as bethe_gf_d1_z
from .lattice.bethe import gf_d2_z as bethe_gf_d2_z
from .lattice.bethe import gf_z as bethe_gf_z
from .lattice.bethe import hilbert_transform as bethe_hilbert_transform

# Body-centered cubic lattice
from .lattice.fcc import dos as fcc_dos
from .lattice.fcc import dos_moment as fcc_dos_moment
from .lattice.fcc import gf_z as fcc_gf_z
from .lattice.fcc import hilbert_transform as fcc_hilbert_transform

# Honeycomb lattice
from .lattice.honeycomb import dos as honeycomb_dos
from .lattice.honeycomb import dos_moment as honeycomb_dos_moment
from .lattice.honeycomb import gf_z as honeycomb_gf_z
from .lattice.honeycomb import hilbert_transform as honeycomb_hilbert_transform

# One-dimensional lattice
from .lattice.onedim import dos as onedim_dos
from .lattice.onedim import dos_moment as onedim_dos_moment
from .lattice.onedim import gf_z as onedim_gf_z
from .lattice.onedim import hilbert_transform as onedim_hilbert_transform

# Simple cubic lattice
from .lattice.sc import dos as sc_dos
from .lattice.sc import dos_moment as sc_dos_moment
from .lattice.sc import gf_z as sc_gf_z
from .lattice.sc import hilbert_transform as sc_hilbert_transform

# Square lattice
from .lattice.square import dos as square_dos
from .lattice.square import dos_moment as square_dos_moment
from .lattice.square import gf_z as square_gf_z
from .lattice.square import hilbert_transform as square_hilbert_transform

# Triangular lattice
from .lattice.triangular import dos as triangular_dos
from .lattice.triangular import dos_moment as triangular_dos_moment
from .lattice.triangular import gf_z as triangular_gf_z
from .lattice.triangular import hilbert_transform as triangular_hilbert_transform

# Fermi and Bose statistics
from .statistics import (
    bose_fct,
    fermi_fct,
    fermi_fct_d1,
    fermi_fct_inv,
    matsubara_frequencies,
    matsubara_frequencies_b,
    pade_frequencies,
)

LOGGER = logging.getLogger(__name__)

# silence warnings of unused imports
assert __version__
assert all((beb, cpa, fourier, lattice, linearprediction, polepade))
assert all((bethe_dos, bethe_dos_moment, bethe_gf_d1_z, bethe_gf_d2_z,
            bethe_gf_z, bethe_hilbert_transform))
assert all((onedim_dos, onedim_dos_moment, onedim_gf_z, onedim_hilbert_transform))
assert all((square_dos, square_dos_moment, square_gf_z, square_hilbert_transform))
assert all((triangular_dos, triangular_dos_moment, triangular_gf_z, triangular_hilbert_transform))
assert all((honeycomb_dos, honeycomb_dos_moment, honeycomb_gf_z, honeycomb_hilbert_transform))
assert all((sc_dos, sc_dos_moment, sc_gf_z, sc_hilbert_transform))
assert all((bcc_dos, bcc_dos_moment, bcc_gf_z, bcc_hilbert_transform))
assert all((fcc_dos, fcc_dos_moment, fcc_gf_z, fcc_hilbert_transform))
assert all((fermi_fct, fermi_fct_d1, fermi_fct_inv, matsubara_frequencies, pade_frequencies))
assert all((bose_fct, matsubara_frequencies_b))
assert all((pole_gf_z, pole_gf_d1_z, pole_gf_tau, pole_gf_ret_t, pole_gf_moments))
assert all((density_iw, chemical_potential))
assert gftool.siam


def surface_gf_zeps(z, eps, hopping_nn):
    r"""
    Surface Green's function for stacked layers.

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
    complex
        Value of the surface Green's function.

    References
    ----------
    .. [odashima2016] Odashima, Mariana M., Beatriz G. Prado, and E. Vernek. Pedagogical
       Introduction to Equilibrium Green's Functions: Condensed-Matter Examples
       with Numerical Implementations. Revista Brasileira de Ensino de Fisica 39,
       no. 1 (September 22, 2016).
       https://doi.org/10.1590/1806-9126-rbef-2016-0087.
    """
    return bethe_gf_z(z-eps, half_bandwidth=2.*hopping_nn)


def hubbard_dimer_gf_z(z, hopping, interaction, kind="+"):
    r"""
    Green's function for the two site Hubbard model on a *dimer*.

    The Hamilton is given

    .. math:: H = -t∑_{σ}(c^†_{1σ} c_{2σ} + c^†_{2σ} c_{1σ}) + U∑_i n_{i↑} n_{i↓}

    with the `hopping` :math:`t` and the `interaction` :math:`U`.
    The Green's function is given for the operators :math:`c_{±σ} = 1/√2 (c_{1σ} ± c_{2σ})`,
    where :math:`±` is given by `kind`

    Parameters
    ----------
    z : complex ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    hopping : float
        The hopping parameter between the sites of the dimer.
    interaction : float
        The Hubbard interaction strength for the on-site interaction.
    kind : {'+', '-'}
        The operator  for which the Green's function is calculated.

    Returns
    -------
    complex ndarray
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
    if kind not in ("+", "-"):
        msg = f"invalid literal for `kind`: '{kind}'"
        raise ValueError(msg)
    s = 1 if kind == "+" else -1
    t = hopping
    U = interaction
    W = (0.25*U*U + 4*t*t)**0.5
    E_0 = 0.5*U - W
    gf_z = (0.5 + s*t/W) / (z - (E_0 + s*t))
    gf_z += (0.5 - s*t/W) / (z - (U + s*t - E_0))
    return gf_z


def hubbard_I_self_z(z, U, occ):
    r"""
    Self-energy in Hubbard-I approximation (atomic solution).

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
    complex array_like
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


def pole_gf_tau_b(tau, poles, weights, beta):
    """
    Bosonic imaginary time Green's function given by a finite number of `poles`.

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
        Inverse temperature.

    Returns
    -------
    (...) float np.ndarray
        Imaginary time Green's function.

    Raises
    ------
    ValueError
        If any `poles.real <= 0`.

    See Also
    --------
    pole_gf_z : Corresponding commutator Green's function.

    Examples
    --------
    >>> beta = 10
    >>> tau = np.linspace(0, beta, num=1000)
    >>> gf_tau = gt.pole_gf_tau_b(tau, .5, 1., beta=beta)

    The integrated imaginary time Green's function gives `-np.sum(weights/poles)`

    >>> np.trapezoid(gf_tau, x=tau)
    np.float64(-2.0000041750107735)

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
        msg = "Bosonic Green's function only well-defined for positive `poles`."
        raise ValueError(msg)
    # eps((beta-tau)*pole)*g(pole, beta) = -exp(-tau*pole)*g(pole, -beta)
    return _gu_sum(weights*bose_fct(poles, -beta)*np.exp(-tau*poles))


Result = namedtuple("Result", ["x", "err"])  # noqa: PYI024


def density(gf_iw, potential, beta, return_err=True, matrix=False, total=False):
    r"""
    Calculate the number density of the Green's function `gf_iw` at finite temperature `beta`.

    .. deprecated:: 0.8.0
       Mostly superseded by more flexible `density_iw`, thus this function will
       likely be discontinued. Currently `density` is a little more accurate
       for `matrix=True`, compared to `density_iw` without using fitting.

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
    matrix : bool, optional
        Whether the given `potential` is a matrix (default: False).
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
       1961): 942-49. https://doi.org/10.1103/PhysRev.121.942.
    """
    warnings.warn("`density` is deprecated; use `density_iw` instead.",
                  category=DeprecationWarning, stacklevel=1)
    iw = matsubara_frequencies(np.arange(gf_iw.shape[-1]), beta=beta)
    if total:
        assert gf_iw.ndim == 1

    if matrix:
        dec = gtmatrix.decompose_her(potential)
        eig = dec.eig
        tail = dec.reconstruct(1./np.add.outer(iw, eig), kind="diag")
        tail = np.moveaxis(tail, source=0, destination=-1)
        analytic = dec.reconstruct(fermi_fct(-eig, beta=beta), kind="diag")
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
        if np.any(err > return_err):
            warnings.warn("density result inaccurate, error estimate = "
                          + str(err), Warning, stacklevel=1)
    return density


def density_error(delta_gf_iw, iw_n, noisy=True):
    """
    Return an estimate for the upper bound of the error in the density.

    This estimate is based on the *integral test*. The crucial assumption is,
    that `ω_N` is large enough, such that :math:`ΔG ~ 1/ω_n^2` for all larger
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
    noisy : bool, optional
        Whether the input `delta_gf_iw` contains noise (default: True).
        If `noisy`, an average over the highest frequency is taken to estimate
        the error.

    Returns
    -------
    float
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
    return factor * denominator


def density_error2(delta_gf_iw, iw_n):
    """
    Return an estimate for the upper bound of the error in the density.

    This estimate is based on the *integral test*. The crucial assumption is,
    that `ω_N` is large enough, such that :math:`ΔG ~ 1/ω_n^3` for all larger
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
    float
        The estimate of the upper bound of the error. Reliable only for large
        enough Matsubara frequencies.
    """
    delta_gf_iw = abs(delta_gf_iw.real)
    part = slice(iw_n.size//10, None, None)  # only consider last 10, iw must be big
    wn = iw_n[part].imag
    denominator = 1./2.*np.pi/wn[-1]**2
    factor = np.max(delta_gf_iw[..., part] * wn**3, axis=-1)
    return factor * denominator


def check_convergence(gf_iw, potential, beta, order=2, matrix=False, total=False):
    """
    Return data for visual inspection of  the density error.

    The calculation of the density error assumes that *sufficient* Matsubara
    frequencies were used. Sufficient means here, that the reminder :math:`ΔG`
    does **not** grow anymore. If the error estimate is small, but
    `check_convergence` returns rapidly growing data, the number of Matsubara
    frequencies is not sufficient

    See `density` for parameters.

    Returns
    -------
    float ndarray
        The last dimension of `check_convergence` corresponds to the Matsubara
        frequencies.

    Other Parameters
    ----------------
    order : int
        The assumed order of the first non-vanishing term of the Laurent expansion.
    """
    iw = matsubara_frequencies(np.arange(gf_iw.shape[-1]), beta=beta)

    if matrix:
        dec = gtmatrix.decompose_her(potential)
        tail = dec.reconstruct(1./np.add.outer(dec.eig, iw), kind="diag")
    else:
        tail = 1/np.add.outer(potential, iw)

    if total:
        tail = tail.real.sum(axis=tuple(range(tail.ndim - 1)))

    delta_g_re = gf_iw.real - tail.real
    return iw.imag**order * delta_g_re
