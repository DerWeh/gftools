# encoding: utf-8
"""Collection of commonly used Green's functions and utilities.

So far mainly contains Bethe Green's functions.
Main purpose is to have a tested base.

Subpackages
-----------
    matrix   --- Work with Green's functions in matrix form, mainly for r-DMFT

.. _Georges et al:
    https://doi.org/10.1103/RevModPhys.68.13
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import warnings
from collections import namedtuple

import numpy as np

from ._version import get_versions

__version__ = get_versions()['version']


def fermi_fct(eps, beta):
    r"""Return the Fermi function :math:`1/(\exp(βz)+1)`.

    Parameters
    ----------
    eps : float or ndarray(float)
        The energy at which the Fermi function is evaluated.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.

    Returns
    -------
    fermi_fct : float or ndarray(float)
        The Fermi function.

    """
    return 0.5 * (1. + np.tanh(-0.5 * beta * eps))


def matsubara_frequencies(n_points, beta):
    r"""Return *fermionic* Matsubara frequencies :math:`iω_n` for the points `n_points`.

    Parameters
    ----------
    n_points : array(int)
        Points for which the Matsubara frequencies :math:`iω_n` are returned.
    beta : float
        Inverse temperature `beta` = 1/T

    Returns
    -------
    matsubara_frequencies : array(complex)
        Array of the imaginary Matsubara frequencies

    """
    return 1j * np.pi * (2*n_points + 1) / beta


def bethe_dos(eps, half_bandwidth):
    """DOS of non-interacting Bethe lattice for infinite coordination number.

    Parameters
    ----------
    eps : array(float), float
        DOS is evaluated at points `eps`.
    half_bandwidth : float
        Half-bandwidth of the DOS, DOS(| `eps` | > `half_bandwidth`) = 0.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`

    Returns
    -------
    result : array(float), float
        The value of the DOS.

    """
    D2 = half_bandwidth * half_bandwidth
    eps2 = eps*eps
    mask = eps2 < D2
    try:
        result = np.empty_like(eps)
        result[~mask] = 0
    except IndexError:  # eps is scalar
        if mask:
            return np.sqrt(D2 - eps2) / (0.5 * np.pi * D2)
        return 0.  # outside of bandwidth
    else:
        result[mask] = np.sqrt(D2 - eps2[mask]) / (0.5 * np.pi * D2)
        return result


def bethe_gf_omega(z, half_bandwidth):
    """Local Green's function of Bethe lattice for infinite Coordination number.

    Parameters
    ----------
    z : array(complex), complex
        Green's function is evaluated at complex frequency `z`
    half_bandwidth : float
        half-bandwidth of the DOS of the Bethe lattice
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`

    Returns
    -------
    bethe_gf_omega : array(complex), complex
        Value of the Green's function

    TODO: source

    """
    z_rel = z / half_bandwidth
    return 2./half_bandwidth*z_rel*(1 - np.sqrt(1 - 1/(z_rel*z_rel)))


def bethe_hilbert_transfrom(xi, half_bandwidth):
    r"""Hilbert transform of non-interacting DOS of the Bethe lattice.

    FIXME: the lattice Hilbert transform is the same as the non-interacting
        Green's function.

    The Hilbert transform

    .. math::
        \tilde{D}(ξ) = ∫_{-∞}^{∞}dϵ \frac{DOS(ϵ)}{ξ − ϵ}

    takes for Bethe lattice in the limit of infinite coordination number the
    explicit form

    .. math::
        \tilde{D}(ξ) = 2*(ξ - s\sqrt{ξ^2 - D^2})/D^2

    with :math:`s=sgn[ℑ{ξ}]`.
    See `Georges et al`_.


    Parameters
    ----------
    xi : array(complex), complex
        Point at which the Hilbert transform is evaluated
    half_bandwidth : float
        half-bandwidth of the DOS of the Bethe lattice

    Returns
    -------
    bethe_hilbert_transfrom : array(complex), complex
        Hilbert transform of `xi`.

    Note
    ----
    Relation between nearest neighbor hopping `t` and half-bandwidth `D`

    .. math::
        2t = D

    """
    return bethe_gf_omega(xi, half_bandwidth)


def bethe_surface_gf(z, eps, hopping_nn):
    r"""Surface Green's function for stacked layers of Bethe lattices.

    .. math::
        \left(1 - \sqrt{1 - 4 t^2 g_{00}^2}\right)/(2 t^2 g_{00})

    with :math:`g_{00} = (z-ϵ)^{-1}`

    TODO: source

    Parameters
    ----------
    z : complex
        Green's function is evaluated at complex frequency `z`.
    eps : float
        Eigenenergy (dispersion) for which the Green's function is evaluated.
    hopping_nn : float
        Nearest neighbor hopping `t` between neighboring layers.

    Returns
    -------
    bethe_surface_gf : complex
        Value of the surface Green's function

    """
    return bethe_gf_omega(z-eps, half_bandwidth=2.*hopping_nn)


def hubbard_dimer_gf_omega(z, hopping, interaction, kind='+'):
    r"""Green's function for the two site Hubbard model on a *dimer*.

    The Hamilton is given

    .. math:: H = -t∑_{σ}(c^†_{1σ} c_{2σ} + c^†_{2σ} c_{1σ}) + U∑_i n_{i↑} n_{i↓}

    with the `hopping` :math:`t` and the `interaction` :math:`U`.
    The Green's function is given for the operators :math:`c_{±σ} = 1/√2 (c_{1σ} ± c_{2σ})`,
    where :math:`±` is given by `kind`

    Parameters
    ----------
    z : array(complex), complex
        Green's function is evaluated at complex frequency `z`
    hopping : float
        The hopping parameter between the sites of the dimer.
    interaction : float
        The Hubbard interaction strength for the on-site interaction.
    kind : {'+', '-'}
        The operator  for which the Green's function is calculated.

    Returns
    -------
    gf_omega : array(complex)
        Value of the Hubbard dimer Green's function at frequencies `z`.

    Notes
    -----
    The solution is obtained by exact digitalization and shown in [4]_.

    References
    ----------
    .. [4] Eder, Robert. “Introduction to the Hubbard Mode.” In The Physics of
       Correlated Insulators, Metals and Superconductors, edited by Eva
       Pavarini, Erik Koch, Richard Scalettar, and Richard Martin. Schriften
       Des Forschungszentrums Jülich Reihe Modeling and Simulation 7. Jülich:
       Forschungszentrum Jülich, 2017.
       https://www.cond-mat.de/events/correl17/manuscripts/eder.pdf.

    """
    if kind not in ('+', '-'):
        raise ValueError("invalid literal for `kind`: '{}'".format(kind))
    s = 1 if kind == '+' else -1
    t = hopping
    U = interaction
    W = (0.25*U*U + 4*t*t)**0.5
    E_0 = 0.5*U - W
    gf_omega  = (0.5 + s*t/W) / (z - (E_0 + s*t))
    gf_omega += (0.5 - s*t/W) / (z - (U + s*t - E_0))
    return gf_omega


Result = namedtuple('Result', ['x', 'err'])


def density(gf_iw, potential, beta, return_err=True):
    r"""Calculate the number density of the Green's function `gf_iw` at finite temperature `beta`.

    As Green's functions decay only as :math:`1/ω`, the known part of the form
    :math:`1/(iω_n + μ - ϵ - ℜΣ_{\text{static}})` will be calculated analytically.
    :math:`Σ_{\text{static}}` is the ω-independent mean-field part of the self-energy.

    Parameters
    ----------
    gf_iw : array(complex)
        The Matsubara frequency Green's function for positive frequencies :math:`iω_n`.
        The last axis corresponds to the Matsubara frequencies.
    potential : float, array(float)
        The static potential for the large-ω behavior of the Green's function.
        It is the real constant :math:`μ - ϵ - ℜΣ_{\text{static}}`.
        The shape must agree with `gf_iw` without the last axis.
    beta : float
        The inverse temperature `beta` = 1/T.
    return_err : bool or float, optional
        If `True` (default), the error estimate will be returned along with the density.
        If `return_err` is a float, a warning will Warning will be issued if
        the error estimate is larger than `return_err`. If `False`, no error
        estimate is calculated.
        See `density_error` for description of the error estimate.

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
    tail = 1/np.add.outer(potential, iw)
    delta_g_re = gf_iw.real - tail.real
    density = 2 * np.sum(delta_g_re, axis=-1) / beta
    density += fermi_fct(-potential, beta=beta)
    if return_err:
        err = density_error(delta_g_re, iw)
        if return_err is True:
            return Result(x=density, err=err)
        else:
            if err > return_err:
                warnings.warn("density result inaccurate, error estimate = "
                              + str(err), Warning)
    return density


def density_error(delta_gf_iw, iw_n):
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
    iw_n : (N) ndarray(complex)
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
    denominator = 1./np.pi/wn[-1]
    factor = np.max(delta_gf_iw[..., part] * wn**2, axis=-1)
    estimate = factor * denominator
    return estimate
