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
      <ϵ, epsilon> (real) energy variable. Typically used for for the :term:`DOS`
      where it replaces the k-dependent Dispersion :math:`ϵ_k`.

   iv
      Bosonic Matsubara frequncies

   iw
      <iω_n> Fermionic Matusbara frequncies

   tau
      Imaginary time points

   z
      Complex frequency variable

"""
import warnings

from functools import partial
from collections import namedtuple

import numpy as np

from scipy.special import expit, logit
from numpy import newaxis

from . import lattice, matrix as gtmatrix
from ._version import get_versions

__version__ = get_versions()['version']


def bose_fct(eps, beta):
    r"""Return the Bose function `1/(exp(βϵ)-1)`.

    Parameters
    ----------
    eps : complex or float or ndarray
        The energy at which the Bose function is evaluated.
        The real part needs to be positive `eps.real > 0`
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.

    Returns
    -------
    bose_fct : complex of float or ndarray
        The Bose function, same type as eps.

    Raises
    ------
    ValueError
        If `eps.real < 0`.

    """
    if np.any(eps.real < 0):
        raise ValueError("Bose function only well defined for non-negative energies `eps`.")
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


# Bethe lattice
bethe_gf_z = lattice.bethe.gf_z
bethe_gf_d1_z = lattice.bethe.gf_d1_z
bethe_gf_d2_z = lattice.bethe.gf_d2_z
bethe_hilbert_transfrom = lattice.bethe.hilbert_transform
bethe_dos = lattice.bethe.dos
bethe_dos.m1 = partial(lattice.bethe.dos_moment, 1)
bethe_dos.m2 = partial(lattice.bethe.dos_moment, 2)
bethe_dos.m3 = partial(lattice.bethe.dos_moment, 3)
bethe_dos.m4 = partial(lattice.bethe.dos_moment, 4)
bethe_dos.m5 = partial(lattice.bethe.dos_moment, 5)

# Square lattice
square_gf_z = lattice.square.gf_z
square_hilbert_transfrom = lattice.square.hilbert_transform
square_dos = lattice.square.dos
square_dos.m1 = partial(lattice.square.dos_moment, 1)
square_dos.m2 = partial(lattice.square.dos_moment, 2)
square_dos.m3 = partial(lattice.square.dos_moment, 3)
square_dos.m4 = partial(lattice.square.dos_moment, 4)
square_dos.m5 = partial(lattice.square.dos_moment, 5)


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
    gf_omega  = (0.5 + s*t/W) / (z - (E_0 + s*t))
    gf_omega += (0.5 - s*t/W) / (z - (U + s*t - E_0))
    return gf_omega


# FIXME: write tests for moments
def hubbard_I_self_z(z, U, occ):
    r"""Self-energy in Hubbard-I approximation (atomic solution).

    The chemical potential and the onsite energy have to be included in `z`.

    Parameters
    ----------
    z : complex array_like
        The complex frequencies at which the self-energy is evaluated. `z`
        should be shifted by the onsite energy and the chemical potential.
    U : float
        The local Hubbard interaction `U`.
    occ : float or float np.ndarray
        The occupation of the opposite spin as the spin of the self-energy.

    Returns
    -------
    Σ_{Hub I} : (\*z.shape, \*occ.shape) complex np.ndarray
        The self-energy in Hubbard I approximation, the shape is the sum of the
        shape of `z` and `occ`.

    """
    hartree = U*occ
    U_1mocc = U*(1 - occ)
    return hartree * (1 + U_1mocc / np.subtract.outer(z, U_1mocc))


def pole_gf_z(z, poles, weights):
    """Green's function given by a finite number of `poles`.

    To be a Green's function, `np.sum(weights)` has to be 1 for the 1/z tail.

    Parameters
    ----------
    z : (...) complex np.ndarray
        Green's function is evaluated at complex frequency `z`.
    poles, weights : (N, ) float np.ndarray
        The position and weight of the poles.

    Returns
    -------
    pole_gf_z : (...) complex np.ndarray
        Green's function shaped like `z`.

    See Also
    --------
    pole_gf_tau : corresponding fermionic imaginary time Green's function
    pole_gf_tau_b : corresponding bosonic imaginary time Green's function

    """
    return np.sum(weights/(np.subtract.outer(z, poles)), axis=-1)


def pole_gf_tau(tau, poles, weights, beta):
    """Imaginary time Green's function given by a finite number of `poles`.

    Parameters
    ----------
    tau : (...) float np.ndarray
        Green's function is evaluated at imaginary times `tau`.
        Only implemented for :math:`τ ∈ [0, β]`.
    poles, weights : (N,) float np.ndarray
        Position and weight of the poles.
    beta : float
        Inverse temperature

    Returns
    -------
    pole_gf_tau : (...) float np.ndarray
        Imaginary time Green's function shaped like `tau`.

    See Also
    --------
    pole_gf_z : corresponding commutator Green's function

    """
    assert np.all((tau >= 0.) & (tau <= beta))
    poles, weights = np.atleast_1d(*np.broadcast_arrays(poles, weights))
    tau = np.asanyarray(tau)
    tau = tau.reshape(tau.shape + (1,)*poles.ndim)
    # exp(-tau*pole)*f(-pole, beta) = exp((beta-tau)*pole)*f(pole, beta)
    exponent = np.where(poles.real >= 0, -tau, beta-tau) * poles
    single_pole_tau = np.exp(exponent) * fermi_fct(-np.sign(poles.real)*poles, beta)
    return -np.sum(weights*single_pole_tau, axis=-1)


def pole_gf_tau_b(tau, poles, weights, beta):
    """Bosonic imaginary time Green's function given by a finite number of `poles`.

    The bosonic Green's function is given by
    `G(tau) = -(1 + bose_fct(poles, beta))*exp(-poles*tau)`

    Parameters
    ----------
    tau : (...) float np.ndarray
        Green's function is evaluated at imaginary times `tau`.
        Only implemented for :math:`τ ∈ [0, β]`.
    poles, weights : (N,) float np.ndarray
        Position and weight of the poles. The real part of the poles needs to
        be positive `poles.real > 0`.
    beta : float
        Inverse temperature

    Returns
    -------
    pole_gf_tau_b : (...) float np.ndarray
        Imaginary time Green's function shaped like `tau`.

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
    poles, weights = np.atleast_1d(*np.broadcast_arrays(poles, weights))
    if np.any(poles.real <= 0):
        raise ValueError("Bosonic Green's function only well-defined for positive `poles`.")
    # eps((beta-tau)*pole)*g(pole, beta) = -exp(-tau*pole)*g(pole, -beta)
    return np.sum(weights*bose_fct(poles, -beta)*np.exp(np.multiply.outer(-tau, poles)), axis=-1)


def pole_gf_moments(poles, weights, order):
    r"""High-frequency moments of the pole Green's function.

    Return the moments `mom` of the expansion :math:`g(z) = \sum_m mom_m/z^m`
    For the pole Green's function we have the simple relation
    :math:`1/(z - ϵ) = \sum_{m=1} ϵ^{m-1}/z^m`.

    Parameters
    ----------
    poles, weights : (..., N) float np.ndarray
        Position and weight of the poles.
    order : (..., M) int array_like
        Order (degree) of the moments.

    Returns
    -------
    mom : (..., M) float np.ndarray
        High-frequency moments.

    """
    poles, weights = np.atleast_1d(*np.broadcast_arrays(poles, weights))
    order = np.asarray(order)[..., newaxis]
    return np.sum(weights[..., newaxis, :] * poles[..., newaxis, :]**(order-1), axis=-1)


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
        tail = dec.reconstruct(1./np.add.outer(xi, iw), kind='diag')
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
