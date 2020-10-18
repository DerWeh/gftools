r"""Fourier transformations of Green's functions.

Fourier transformation between imaginary time and Matsubara frequencies.
The function in this module should be used after explicitly treating the
high-frequency behavior, as this is not yet implemented.
Typically, transformation from τ-space to Matsubara frequency are unproblematic.

The Fourier transforms are defined in the following way:

Definitions
-----------

imaginary time → Matsubara frequencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Fourier integral for the Matsubara Green's function is defined as:

.. math:: G(iw_n) = 0.5 ∫_{-β}^{β}dτ G(τ) \exp(iw_n τ)

with :math:`iw_n = iπn/β`. For fermionic Green's functions only odd frequencies
are non-vanishing, for bosonic Green's functions only even.

The recommended high-level function to perform this Fourier transform is:

* `tau2iw` for *fermionic* Green's functions
* `tau2iv` for *bosonic* Green's functions

Matsubara frequencies → imaginary time
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Fourier sum for the imaginary time Green's function is defined as:

.. math:: G(τ) = 1/β \sum_{n=-\infty}^{\infty} G(iw_n) \exp(-iw_n τ).

The recommended high-level function to perform this Fourier transform is:

* `iw2tau` for *fermionic* Green's functions

Glossary
--------

.. glossary::

   dft
      <discrete Foruier transform>

   ft
      <Fourier transformation> In contrast to :term:`dft`, this is used for
      Fourier integration of continous variables without discretization.

Previously defined:

* :term:`iv`
* :term:`iw`
* :term:`tau`

"""
import logging

import numpy as np
from numpy import newaxis

import gftool as gt
from gftool.basis.pole import PoleGf

LOGGER = logging.getLogger(__name__)


def iw2tau_dft(gf_iw, beta):
    r"""Discrete Fourier transform of the Hermitian Green's function `gf_iw`.

    Fourier transformation of a fermionic Matsubara Green's function to
    imaginary-time domain.
    The infinite Fourier sum is truncated.
    We assume a Hermitian Green's function `gf_iw`, i.e. :math:`G(-iω_n) = G^*(iω_n)`,
    which is the case for commutator Green's functions :math:`G_{AB}(τ) = ⟨A(τ)B⟩`
    with :math:`A = B^†`. The Fourier transform `gf_tau` is then real.

    Parameters
    ----------
    gf_iw : (..., N_iw) complex np.ndarray
        The Green's function at positive **fermionic** Matsubara frequencies
        :math:`iω_n`.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.

    Returns
    -------
    gf_tau : (..., 2*N_iw + 1) float np.ndarray
        The Fourier transform of `gf_iw` for imaginary times :math:`τ \in [0, β]`.

    See Also
    --------
    iw2tau_dft_soft : Fourier transform with artificial softening of oszillations

    Notes
    -----
    For accurate an accurate Fourier transform, it is necessary, that `gf_iw`
    has already reached it's high-frequency behaviour, which need to be included
    explicitly. Therefore, the accuracy of the FT depends implicitely on the
    bandwidth!

    Examples
    --------
    >>> import gftool.fourier
    >>> BETA = 50
    >>> iws = gt.matsubara_frequencies(range(1024), beta=BETA)
    >>> tau = np.linspace(0, BETA, num=2*iws.size + 1, endpoint=True)

    >>> poles = 2*np.random.random(10) - 1  # partially filled
    >>> weights = np.random.random(10)
    >>> weights = weights/np.sum(weights)
    >>> gf_iw = gt.pole_gf_z(iws, poles=poles, weights=weights)
    >>> # 1/z tail has to be handled manually
    >>> gf_dft = gt.fourier.iw2tau_dft(gf_iw - 1/iws, beta=BETA) - .5
    >>> gf_iw.size, gf_dft.size
    (1024, 2049)
    >>> gf_tau = gt.pole_gf_tau(tau, poles=poles, weights=weights, beta=BETA)

    >>> import matplotlib.pyplot as plt
    >>> __ = plt.plot(tau, gf_tau, label='exact')
    >>> __ = plt.plot(tau, gf_dft, '--', label='DFT')
    >>> __ = plt.legend()
    >>> plt.show()

    >>> __ = plt.title('Oscillations around boundaries 0, β')
    >>> __ = plt.plot(tau/BETA, gf_tau - gf_dft)
    >>> __ = plt.xlabel('τ/β')
    >>> plt.show()

    The method is resistant against noise:

    >>> magnitude = 2e-7
    >>> noise = np.random.normal(scale=magnitude, size=gf_iw.size)
    >>> gf_dft_noisy = gt.fourier.iw2tau_dft(gf_iw + noise - 1/iws, beta=BETA) - .5
    >>> __ = plt.plot(tau, abs(gf_tau - gf_dft_noisy), '--', label='noisy')
    >>> __ = plt.axhline(magnitude, color='black')
    >>> __ = plt.plot(tau, abs(gf_tau - gf_dft), label='clean')
    >>> __ = plt.legend()
    >>> plt.yscale('log')
    >>> plt.show()

    """
    gf_iwall = np.zeros(gf_iw.shape[:-1] + (2*gf_iw.shape[-1] + 1,), dtype=gf_iw.dtype)
    gf_iwall[..., 1:-1:2] = gf_iw  # GF containing fermionic and bosonic Matsubaras
    gf_tau = np.fft.hfft(1./beta * gf_iwall)
    gf_tau = gf_tau[..., :gf_iwall.shape[-1]]  # trim to tau in [0, beta]  # pylint: disable=unsubscriptable-object,C0301
    return gf_tau


def iw2tau_dft_soft(gf_iw, beta):
    r"""Discrete Fourier transform of the Hermitian Green's function `gf_iw`.

    Fourier transformation of a fermionic Matsubara Green's function to
    imaginary-time domain.
    Add a tail letting `gf_iw` go to 0. The tail is just a cosine function to
    exactly hit the 0.
    This is unphysical but suppresses oscillations. This methods should be used
    with care, as it might hide errors.
    We assume a Hermitian Green's function `gf_iw`, i.e. :math:`G(-iω_n) = G^*(iω_n)`,
    which is the case for commutator Green's functions :math:`G_{AB}(τ) = ⟨A(τ)B⟩`
    with :math:`A = B^†`. The Fourier transform `gf_tau` is then real.

    Parameters
    ----------
    gf_iw : (..., N_iw) complex np.ndarray
        The Green's function at positive **fermionic** Matsubara frequencies
        :math:`iω_n`.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.

    Returns
    -------
    gf_tau : (..., 2*N_iw + 1) float np.ndarray
        The Fourier transform of `gf_iw` for imaginary times :math:`τ \in [0, β]`.

    See Also
    --------
    iw2tau_dft : Plain implementation of Fourier transform

    Notes
    -----
    For accurate an accurate Fourier transform, it is necessary, that `gf_iw`
    has already reached it's high-frequency behaviour, which need to be included
    explicitly. Therefore, the accuracy of the FT depends implicitely on the
    bandwidth!

    Examples
    --------
    >>> import gftool.fourier
    >>> BETA = 50
    >>> iws = gt.matsubara_frequencies(range(1024), beta=BETA)
    >>> tau = np.linspace(0, BETA, num=2*iws.size + 1, endpoint=True)

    >>> poles = 2*np.random.random(10) - 1  # partially filled
    >>> weights = np.random.random(10)
    >>> weights = weights/np.sum(weights)
    >>> gf_iw = gt.pole_gf_z(iws, poles=poles, weights=weights)
    >>> # 1/z tail has to be handled manually
    >>> gf_dft = gt.fourier.iw2tau_dft_soft(gf_iw - 1/iws, beta=BETA) - .5
    >>> gf_iw.size, gf_dft.size
    (1024, 2049)
    >>> gf_tau = gt.pole_gf_tau(tau, poles=poles, weights=weights, beta=BETA)

    >>> import matplotlib.pyplot as plt
    >>> __ = plt.plot(tau, gf_tau, label='exact')
    >>> __ = plt.plot(tau, gf_dft, '--', label='DFT')
    >>> __ = plt.legend()
    >>> plt.show()

    >>> __ = plt.title('Oscillations around boundaries 0, β slightly suppressed')
    >>> __ = plt.plot(tau/BETA, gf_tau - gf_dft, label='DFT soft')
    >>> gf_dft_bare = gt.fourier.iw2tau_dft(gf_iw - 1/iws, beta=BETA) - .5
    >>> __ = plt.plot(tau/BETA, gf_tau - gf_dft_bare, '--',  label='DFT bare')
    >>> __ = plt.legend()
    >>> __ = plt.xlabel('τ/β')
    >>> plt.show()

    The method is resistant against noise:

    >>> magnitude = 2e-7
    >>> noise = np.random.normal(scale=magnitude, size=gf_iw.size)
    >>> gf_dft_noisy = gt.fourier.iw2tau_dft_soft(gf_iw + noise - 1/iws, beta=BETA) - .5
    >>> __ = plt.plot(tau, abs(gf_tau - gf_dft_noisy), '--', label='noisy')
    >>> __ = plt.axhline(magnitude, color='black')
    >>> __ = plt.plot(tau, abs(gf_tau - gf_dft), label='clean')
    >>> __ = plt.legend()
    >>> plt.yscale('log')
    >>> plt.show()

    """
    tail_range = np.linspace(0, np.pi, num=gf_iw.shape[-1] + 1)[1:]
    tail = .5*(np.cos(tail_range) + 1.)
    LOGGER.debug("Remaining tail approximated by 'cos': %s", gf_iw[..., -1:])
    gf_iw_extended = np.concatenate((gf_iw, tail*gf_iw[..., -1:]), axis=-1)
    gf_tau = iw2tau_dft(gf_iw_extended, beta=beta)[..., ::2]  # trim artificial resolution
    return gf_tau


def iw2tau(gf_iw, beta, moments=(1.,), fourier=iw2tau_dft, n_fit=0):
    r"""Discrete Fourier transform of the Hermitian Green's function `gf_iw`.

    Fourier transformation of a fermionic Matsubara Green's function to
    imaginary-time domain.
    We assume a Hermitian Green's function `gf_iw`, i.e. :math:`G(-iω_n) = G^*(iω_n)`,
    which is the case for commutator Green's functions :math:`G_{AB}(τ) = ⟨A(τ)B⟩`
    with :math:`A = B^†`. The Fourier transform `gf_tau` is then real.

    Parameters
    ----------
    gf_iw : (N_iw) complex np.ndarray
        The Green's function at positive **fermionic** Matsubara frequencies
        :math:`iω_n`.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.
    moments : (m) float array_like
        High-frequency moments of `gf_iw`.
    fourier : {`iw2tau_dft`, `iw2tau_dft_soft`}, optional
        Back-end to perform the actual Fourier transformation.
    n_fit : int, optional
        Number of additionally fitted moments (in fact, `gf_iw` is fitted, not
        not directly moments).

    Returns
    -------
    gf_tau : (2*N_iw + 1) float np.ndarray
        The Fourier transform of `gf_iw` for imaginary times :math:`τ \in [0, β]`.

    See Also
    --------
    iw2tau_dft : Back-end: plain implementation of Fourier transform
    iw2tau_dft_soft : Back-end: Fourier transform with artificial softening of oszillations

    pole_gf_from_moments : Function handling the given `moments`

    Notes
    -----
    For accurate an accurate Fourier transform, it is necessary, that `gf_iw`
    has already reached it's high-frequency behaviour, which need to be included
    explicitly. Therefore, the accuracy of the FT depends implicitely on the
    bandwidth!

    Examples
    --------
    >>> import gftool.fourier
    >>> BETA = 50
    >>> iws = gt.matsubara_frequencies(range(1024), beta=BETA)
    >>> tau = np.linspace(0, BETA, num=2*iws.size + 1, endpoint=True)

    >>> poles = 2*np.random.random(10) - 1  # partially filled
    >>> weights = np.random.random(10)
    >>> weights = weights/np.sum(weights)
    >>> gf_iw = gt.pole_gf_z(iws, poles=poles, weights=weights)
    >>> gf_dft = gt.fourier.iw2tau(gf_iw, beta=BETA)
    >>> gf_iw.size, gf_dft.size
    (1024, 2049)
    >>> gf_tau = gt.pole_gf_tau(tau, poles=poles, weights=weights, beta=BETA)

    >>> import matplotlib.pyplot as plt
    >>> __ = plt.plot(tau, gf_tau, label='exact')
    >>> __ = plt.plot(tau, gf_dft, '--', label='FT')
    >>> __ = plt.legend()
    >>> plt.show()

    >>> __ = plt.title('Oscillations around boundaries 0, β')
    >>> __ = plt.plot(tau/BETA, gf_tau - gf_dft)
    >>> __ = plt.xlabel('τ/β')
    >>> plt.show()

    Results can be drastically improved giving high-frequency moments,
    this reduces the truncation error.

    >>> mom = np.sum(weights[:, np.newaxis] * poles[:, np.newaxis]**range(8), axis=0)
    >>> for n in range(1, 8):
    ...     gf = gt.fourier.iw2tau(gf_iw, moments=mom[:n], beta=BETA)
    ...     __ = plt.plot(tau/BETA, abs(gf_tau - gf), label=f'n_mom={n}')
    >>> __ = plt.legend()
    >>> __ = plt.xlabel('τ/β')
    >>> plt.yscale('log')
    >>> plt.show()

    The method is resistant against noise:

    >>> magnitude = 2e-7
    >>> noise = np.random.normal(scale=magnitude, size=gf_iw.size)
    >>> for n in range(1, 7, 2):
    ...     gf = gt.fourier.iw2tau(gf_iw+noise, moments=mom[:n], beta=BETA)
    ...     __ = plt.plot(tau/BETA, abs(gf_tau - gf), '--', label=f'n_mom={n}')
    >>> __ = plt.axhline(magnitude, color='black')
    >>> __ = plt.plot(tau/BETA, abs(gf_tau - gf_dft), label='clean')
    >>> __ = plt.legend()
    >>> plt.yscale('log')
    >>> plt.show()

    """
    moments = np.asarray(moments)
    iws = gt.matsubara_frequencies(range(gf_iw.shape[-1]), beta=beta)
    # newaxis in pole_gf inserts axis for iws/tau
    if n_fit:
        n_mom = moments.shape[-1]
        pole_gf = PoleGf.from_z(iws, gf_iw[..., newaxis, :], n_pole=n_fit+n_mom,
                                moments=moments[..., newaxis, :], weight=iws.imag**(n_mom+n_fit))
    else:
        pole_gf = PoleGf.from_moments(moments[..., newaxis, :])
    gf_iw = gf_iw - pole_gf.eval_z(iws)
    gf_tau = fourier(gf_iw, beta=beta)
    tau = np.linspace(0, beta, num=gf_tau.shape[-1])
    gf_tau += pole_gf.eval_tau(tau, beta=beta)
    return gf_tau


def tau2iv_dft(gf_tau, beta):
    r"""Discrete Fourier transform of the real Green's function `gf_tau`.

    Fourier transformation of a bosonic imaginary-time Green's function to
    Matsubara domain.
    The Fourier integral is replaced by a Riemann sum giving a discrete
    Fourier transform (DFT).
    We assume a real Green's function `gf_tau`, which is the case for
    commutator Green's functions :math:`G_{AB}(τ) = ⟨A(τ)B⟩` with
    :math:`A = B^†`. The Fourier transform `gf_iv` is then Hermitian.

    Parameters
    ----------
    gf_tau : (..., N_tau) float np.ndarray
        The Green's function at imaginary times :math:`τ \in [0, β]`.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.

    Returns
    -------
    gf_iv : (..., {N_iv + 1}/2) float np.ndarray
        The Fourier transform of `gf_tau` for non-negative bosonic Matsubara
        frequencies :math:`iν_n`.

    See Also
    --------
    tau2iv_ft_lin : Fourier integration using Filon's method

    Examples
    --------
    >>> import gftool.fourier
    >>> BETA = 50
    >>> tau = np.linspace(0, BETA, num=2049, endpoint=True)
    >>> ivs = gt.matsubara_frequencies_b(range((tau.size+1)//2), beta=BETA)

    >>> poles, weights = np.random.random(10), np.random.random(10)
    >>> weights = weights/np.sum(weights)
    >>> gf_tau = gt.pole_gf_tau(tau, poles=poles, weights=weights, beta=BETA)
    >>> gf_dft = gt.fourier.tau2iv_dft(gf_tau, beta=BETA)
    >>> gf_tau.size, gf_dft.size
    (2049, 1025)
    >>> gf_iv = gt.pole_gf_z(ivs, poles=poles, weights=weights)

    >>> import matplotlib.pyplot as plt
    >>> __ = plt.plot(gf_iv.imag, label='exact Im')
    >>> __ = plt.plot(gf_dft.imag, '--', label='DFT Im')
    >>> __ = plt.plot(gf_iv.real, label='exact Re')
    >>> __ = plt.plot(gf_dft.real, '--', label='DFT Re')
    >>> __ = plt.legend()
    >>> plt.show()

    >>> __ = plt.title('Error growing with frequency')
    >>> __ = plt.plot(abs(gf_iv - gf_dft))
    >>> plt.yscale('log')
    >>> plt.show()

    The method is resistant against noise:

    >>> magnitude = 2e-3
    >>> noise = np.random.normal(scale=magnitude, size=gf_tau.size)
    >>> gf_dft_noisy = gt.fourier.tau2iv_dft(gf_tau + noise, beta=BETA)
    >>> __ = plt.plot(abs(gf_iv - gf_dft_noisy), '--', label='noisy')
    >>> __ = plt.axhline(magnitude, color='black')
    >>> __ = plt.plot(abs(gf_iv - gf_dft), label='clean')
    >>> __ = plt.legend()
    >>> # plt.yscale('log')
    >>> plt.show()

    """
    gf_mean = np.trapz(gf_tau, dx=beta/(gf_tau.shape[-1]-1), axis=-1)
    gf_iv = beta * np.fft.ihfft(gf_tau[..., :-1] - gf_mean[..., newaxis])
    gf_iv[..., 0] = gf_mean
    # gives better results in practice but is wrong...
    # gf_iv = beta * np.fft.ihfft(.5*(gf_tau[..., 1:] + gf_tau[..., :-1]))
    return gf_iv


def tau2iv_ft_lin(gf_tau, beta):
    r"""Fourier integration of the real Green's function `gf_tau`.

    Fourier transformation of a bosonic imaginary-time Green's function to
    Matsubara domain.
    We assume a real Green's function `gf_tau`, which is the case for
    commutator Green's functions :math:`G_{AB}(τ) = ⟨A(τ)B⟩` with
    :math:`A = B^†`. The Fourier transform `gf_iv` is then Hermitian.
    Filon's method is used to calculated the Fourier integral

    .. math:: G^n = ∫_{0}^{β}dτ G(τ) e^{iν_n τ},

    :math:`G(τ)` is approximated by a linear spline. A linear approximation was
    chosen to be able to integrate noisy functions. Information on oscillatory
    integrations can be found e.g. in [filon1928]_ and [iserles]_.

    Parameters
    ----------
    gf_tau : (..., N_tau) float np.ndarray
        The Green's function at imaginary times :math:`τ \in [0, β]`.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.

    Returns
    -------
    gf_iv : (..., {N_iv + 1}/2) float np.ndarray
        The Fourier transform of `gf_tau` for non-negative bosonic Matsubara
        frequencies :math:`iν_n`.

    See Also
    --------
    tau2iv_dft : Plain implementation using Riemann sum.

    References
    ----------
    .. [filon1928] L.N. Filon, On a quadrature formula for trigonometric integrals,
       Proc. Roy. Soc. Edinburgh 49 (1928) 38-47.
    .. [iserles] A. Iserles, S.P. Nørsett, and S. Olver, Highly oscillatory
       quadrature: The story so far,
       http://www.sam.math.ethz.ch/~hiptmair/Seminars/OSCINT/INO06.pdf

    Examples
    --------
    >>> import gftool.fourier
    >>> BETA = 50
    >>> tau = np.linspace(0, BETA, num=2049, endpoint=True)
    >>> ivs = gt.matsubara_frequencies_b(range((tau.size+1)//2), beta=BETA)

    >>> poles, weights = np.random.random(10), np.random.random(10)
    >>> weights = weights/np.sum(weights)
    >>> gf_tau = gt.pole_gf_tau_b(tau, poles=poles, weights=weights, beta=BETA)
    >>> gf_ft_lin = gt.fourier.tau2iv_ft_lin(gf_tau, beta=BETA)
    >>> gf_tau.size, gf_ft_lin.size
    (2049, 1025)
    >>> gf_iv = gt.pole_gf_z(ivs, poles=poles, weights=weights)

    >>> import matplotlib.pyplot as plt
    >>> __ = plt.plot(gf_iv.imag, label='exact Im')
    >>> __ = plt.plot(gf_ft_lin.imag, '--', label='DFT Im')
    >>> __ = plt.plot(gf_iv.real, label='exact Re')
    >>> __ = plt.plot(gf_ft_lin.real, '--', label='DFT Re')
    >>> __ = plt.legend()
    >>> plt.show()

    >>> __ = plt.title('Error decreasing with frequency')
    >>> __ = plt.plot(abs(gf_iv - gf_ft_lin), label='FT_lin')
    >>> gf_dft = gt.fourier.tau2iv_dft(gf_tau, beta=BETA)
    >>> __ = plt.plot(abs(gf_iv - gf_dft), '--', label='DFT')
    >>> __ = plt.legend()
    >>> plt.yscale('log')
    >>> plt.show()

    The method is resistant against noise:

    >>> magnitude = 5e-6
    >>> noise = np.random.normal(scale=magnitude, size=gf_tau.size)
    >>> gf_ft_noisy = gt.fourier.tau2iv_ft_lin(gf_tau + noise, beta=BETA)
    >>> __ = plt.plot(abs(gf_iv - gf_ft_noisy), '--', label='noisy')
    >>> __ = plt.axhline(magnitude, color='black')
    >>> __ = plt.plot(abs(gf_iv - gf_ft_lin), label='clean')
    >>> __ = plt.legend()
    >>> plt.yscale('log')
    >>> plt.show()

    """
    n_tau = gf_tau.shape[-1]
    gf_dft = np.fft.ihfft(gf_tau[..., :-1])
    d_gf_dft = np.fft.ihfft(gf_tau[..., 1:] - gf_tau[..., :-1])
    d_tau_ivs = 2j*np.pi/(n_tau - 1)*np.arange(gf_dft.shape[-1])
    d_tau_ivs[..., 0] = np.nan  # avoid zero division, fix value by hand
    expm1 = np.expm1(d_tau_ivs)
    weight1 = expm1/d_tau_ivs
    weight2 = (expm1 + 1 - weight1)/d_tau_ivs
    weight1[0], weight2[0] = 1, .5  # special case n=0
    gf_iv = weight1*gf_dft + weight2*d_gf_dft
    gf_iv = beta*gf_iv
    return gf_iv


def tau2iv(gf_tau, beta, fourier=tau2iv_ft_lin):
    r"""Fourier transformation of the real Green's function `gf_tau`.

    Fourier transformation of a bosonic imaginary-time Green's function to
    Matsubara domain.
    We assume a real Green's function `gf_tau`, which is the case for
    commutator Green's functions :math:`G_{AB}(τ) = ⟨A(τ)B⟩` with
    :math:`A = B^†`. The Fourier transform `gf_iv` is then Hermitian.
    This function removes the discontinuity :math:`G_{AB}(β) - G_{AB}(0) = ⟨[A,B]⟩`.

    TODO: if high-frequency moments are know, they should be stripped for
    increased accuracy.

    Parameters
    ----------
    gf_tau : (..., N_tau) float np.ndarray
        The Green's function at imaginary times :math:`τ \in [0, β]`.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.
    fourier : {`tau2iv_ft_lin`, `tau2iv_dft`}, optional
        Back-end to perform the actual Fourier transformation.

    Returns
    -------
    gf_iv : (..., {N_iv + 1}/2) complex np.ndarray
        The Fourier transform of `gf_tau` for non-negative bosonic Matsubara
        frequencies :math:`iν_n`.

    See Also
    --------
    tau2iv_dft : Back-end: plain implementation using Riemann sum.
    tau2iv_ft_lin : Back-end: Fourier integration using Filon's method.

    Examples
    --------
    >>> import gftool.fourier
    >>> BETA = 50
    >>> tau = np.linspace(0, BETA, num=2049, endpoint=True)
    >>> ivs = gt.matsubara_frequencies_b(range((tau.size+1)//2), beta=BETA)

    >>> poles, weights = np.random.random(10), np.random.random(10)
    >>> weights = weights/np.sum(weights)
    >>> gf_tau = gt.pole_gf_tau_b(tau, poles=poles, weights=weights, beta=BETA)
    >>> gf_ft = gt.fourier.tau2iv(gf_tau, beta=BETA)
    >>> gf_tau.size, gf_ft.size
    (2049, 1025)
    >>> gf_iv = gt.pole_gf_z(ivs, poles=poles, weights=weights)

    >>> import matplotlib.pyplot as plt
    >>> __ = plt.plot(gf_iv.imag, label='exact Im')
    >>> __ = plt.plot(gf_ft.imag, '--', label='DFT Im')
    >>> __ = plt.plot(gf_iv.real, label='exact Re')
    >>> __ = plt.plot(gf_ft.real, '--', label='DFT Re')
    >>> __ = plt.legend()
    >>> plt.show()

    Accuracy of the different back-ends

    >>> ft_lin, dft = gt.fourier.tau2iv_ft_lin, gt.fourier.tau2iv_dft
    >>> gf_ft_lin = gt.fourier.tau2iv(gf_tau, beta=BETA, fourier=ft_lin)
    >>> gf_dft = gt.fourier.tau2iv(gf_tau, beta=BETA, fourier=dft)
    >>> __ = plt.plot(abs(gf_iv - gf_ft_lin), label='FT_lin')
    >>> __ = plt.plot(abs(gf_iv - gf_dft), '--', label='DFT')
    >>> __ = plt.legend()
    >>> plt.yscale('log')
    >>> plt.show()

    The methods are resistant against noise:

    >>> magnitude = 5e-6
    >>> noise = np.random.normal(scale=magnitude, size=gf_tau.size)
    >>> gf_ft_lin_noisy = gt.fourier.tau2iv(gf_tau + noise, beta=BETA, fourier=ft_lin)
    >>> gf_dft_noisy = gt.fourier.tau2iv(gf_tau + noise, beta=BETA, fourier=dft)
    >>> __ = plt.plot(abs(gf_iv - gf_ft_lin_noisy), '--', label='FT_lin')
    >>> __ = plt.plot(abs(gf_iv - gf_dft_noisy), '--', label='DFT')
    >>> __ = plt.axhline(magnitude, color='black')
    >>> __ = plt.legend()
    >>> plt.yscale('log')
    >>> plt.show()

    """
    g1 = (gf_tau[..., -1] - gf_tau[..., 0])  # = 1/z moment = jump of Gf at 0^{±}
    tau = np.linspace(0, beta, num=gf_tau.shape[-1])
    gf_tau = gf_tau - g1[..., newaxis]/beta*tau  # remove jump by linear shift
    gf_iv = fourier(gf_tau, beta=beta)
    ivs = gt.matsubara_frequencies_b(range(1, gf_iv.shape[-1]), beta=beta)
    gf_iv[..., 1:] += g1[..., newaxis]/ivs
    gf_iv[..., 0] += .5 * g1 * beta  # `iv_{n=0}` = 0 has to be treated separately
    return gf_iv


def tau2iw_dft(gf_tau, beta):
    r"""Discrete Fourier transform of the real Green's function `gf_tau`.

    Fourier transformation of a fermionic imaginary-time Green's function to
    Matsubara domain.
    The Fourier integral is replaced by a Riemann sum giving a discrete
    Fourier transform (DFT).
    We assume a real Green's function `gf_tau`, which is the case for
    commutator Green's functions :math:`G_{AB}(τ) = ⟨A(τ)B⟩` with
    :math:`A = B^†`. The Fourier transform `gf_iw` is then Hermitian.

    Parameters
    ----------
    gf_tau : (..., N_tau) float np.ndarray
        The Green's function at imaginary times :math:`τ \in [0, β]`.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.

    Returns
    -------
    gf_iw : (..., {N_iw - 1}/2) float np.ndarray
        The Fourier transform of `gf_tau` for positive fermionic Matsubara
        frequencies :math:`iω_n`.

    See Also
    --------
    tau2iw_ft_lin : Fourier integration using Filon's method

    Examples
    --------
    >>> import gftool.fourier
    >>> BETA = 50
    >>> tau = np.linspace(0, BETA, num=2049, endpoint=True)
    >>> iws = gt.matsubara_frequencies(range((tau.size-1)//2), beta=BETA)

    >>> poles = 2*np.random.random(10) - 1  # partially filled
    >>> weights = np.random.random(10)
    >>> weights = weights/np.sum(weights)
    >>> gf_tau = gt.pole_gf_tau(tau, poles=poles, weights=weights, beta=BETA)
    >>> # 1/z tail has to be handled manually
    >>> gf_dft = gt.fourier.tau2iw_dft(gf_tau + .5, beta=BETA) + 1/iws
    >>> gf_tau.size, gf_dft.size
    (2049, 1024)
    >>> gf_iw = gt.pole_gf_z(iws, poles=poles, weights=weights)

    >>> import matplotlib.pyplot as plt
    >>> __ = plt.plot(gf_iw.imag, label='exact Im')
    >>> __ = plt.plot(gf_dft.imag, '--', label='DFT Im')
    >>> __ = plt.plot(gf_iw.real, label='exact Re')
    >>> __ = plt.plot(gf_dft.real, '--', label='DFT Re')
    >>> __ = plt.legend()
    >>> plt.show()

    >>> __ = plt.title('Error growing with frequency')
    >>> __ = plt.plot(abs(gf_iw - gf_dft))
    >>> plt.yscale('log')
    >>> plt.show()

    The method is resistant against noise:

    >>> magnitude = 2e-5
    >>> noise = np.random.normal(scale=magnitude, size=gf_tau.size)
    >>> gf_dft_noisy = gt.fourier.tau2iw_dft(gf_tau + noise + .5, beta=BETA) + 1/iws
    >>> __ = plt.plot(abs(gf_iw - gf_dft_noisy), '--', label='noisy')
    >>> __ = plt.axhline(magnitude, color='black')
    >>> __ = plt.plot(abs(gf_iw - gf_dft), label='clean')
    >>> __ = plt.legend()
    >>> plt.yscale('log')
    >>> plt.show()

    """
    # expand `gf_tau` to [-β, β] to get symmetric function
    gf_tau_full_range = np.concatenate((-gf_tau[..., :-1], gf_tau), axis=-1)
    dft = np.fft.ihfft(gf_tau_full_range[..., :-1])
    gf_iw = -beta * dft[..., 1::2]  # select *fermionic* Matsubara frequencies
    return gf_iw


def tau2iw_ft_lin(gf_tau, beta):
    r"""Fourier integration of the real Green's function `gf_tau`.

    Fourier transformation of a fermionic imaginary-time Green's function to
    Matsubara domain.
    We assume a real Green's function `gf_tau`, which is the case for
    commutator Green's functions :math:`G_{AB}(τ) = ⟨A(τ)B⟩` with
    :math:`A = B^†`. The Fourier transform `gf_iw` is then Hermitian.
    Filon's method is used to calculated the Fourier integral

    .. math:: G^n = 0.5 ∫_{-β}^{β}dτ G(τ) e^{iω_n τ},

    :math:`G(τ)` is approximated by a linear spline. A linear approximation was
    chosen to be able to integrate noisy functions. Information on oscillatory
    integrations can be found e.g. in [filon1928]_ and [iserles]_.

    Parameters
    ----------
    gf_tau : (..., N_tau) float np.ndarray
        The Green's function at imaginary times :math:`τ \in [0, β]`.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.

    Returns
    -------
    gf_iw : (..., {N_iw - 1}/2) float np.ndarray
        The Fourier transform of `gf_tau` for positive fermionic Matsubara
        frequencies :math:`iω_n`.

    See Also
    --------
    tau2iw_dft : Plain implementation using Riemann sum.

    References
    ----------
    .. [filon1928] L.N. Filon, On a quadrature formula for trigonometric integrals,
       Proc. Roy. Soc. Edinburgh 49 (1928) 38-47.
    .. [iserles] A. Iserles, S.P. Nørsett, and S. Olver, Highly oscillatory
       quadrature: The story so far,
       http://www.sam.math.ethz.ch/~hiptmair/Seminars/OSCINT/INO06.pdf

    Examples
    --------
    >>> import gftool.fourier
    >>> BETA = 50
    >>> tau = np.linspace(0, BETA, num=2049, endpoint=True)
    >>> iws = gt.matsubara_frequencies(range((tau.size-1)//2), beta=BETA)

    >>> poles = 2*np.random.random(10) - 1  # partially filled
    >>> weights = np.random.random(10)
    >>> weights = weights/np.sum(weights)
    >>> gf_tau = gt.pole_gf_tau(tau, poles=poles, weights=weights, beta=BETA)
    >>> # 1/z tail has to be handled manually
    >>> gf_ft_lin = gt.fourier.tau2iw_ft_lin(gf_tau + .5, beta=BETA) + 1/iws
    >>> gf_tau.size, gf_ft_lin.size
    (2049, 1024)
    >>> gf_iw = gt.pole_gf_z(iws, poles=poles, weights=weights)

    >>> import matplotlib.pyplot as plt
    >>> __ = plt.plot(gf_iw.imag, label='exact Im')
    >>> __ = plt.plot(gf_ft_lin.imag, '--', label='DFT Im')
    >>> __ = plt.plot(gf_iw.real, label='exact Re')
    >>> __ = plt.plot(gf_ft_lin.real, '--', label='DFT Re')
    >>> __ = plt.legend()
    >>> plt.show()

    >>> __ = plt.title('Error decreasing with frequency')
    >>> __ = plt.plot(abs(gf_iw - gf_ft_lin), label='FT_lin')
    >>> gf_dft = gt.fourier.tau2iw_dft(gf_tau + .5, beta=BETA) + 1/iws
    >>> __ = plt.plot(abs(gf_iw - gf_dft), '--', label='DFT')
    >>> __ = plt.legend()
    >>> plt.yscale('log')
    >>> plt.show()

    The method is resistant against noise:

    >>> magnitude = 5e-6
    >>> noise = np.random.normal(scale=magnitude, size=gf_tau.size)
    >>> gf_ft_noisy = gt.fourier.tau2iw_ft_lin(gf_tau + noise + .5, beta=BETA) + 1/iws
    >>> __ = plt.plot(abs(gf_iw - gf_ft_noisy), '--', label='noisy')
    >>> __ = plt.axhline(magnitude, color='black')
    >>> __ = plt.plot(abs(gf_iw - gf_ft_lin), label='clean')
    >>> __ = plt.legend()
    >>> plt.yscale('log')
    >>> plt.show()

    """
    gf_tau_full_range = np.concatenate((-gf_tau[..., :-1], gf_tau), axis=-1)
    n_tau = gf_tau_full_range.shape[-1] - 1  # pylint: disable=unsubscriptable-object
    gf_dft = np.fft.ihfft(gf_tau_full_range[..., :-1])
    d_gf_tau = gf_tau_full_range[..., 1:] - gf_tau_full_range[..., :-1]
    d_gf_dft = np.fft.ihfft(d_gf_tau)
    d_tau_iws = 2j*np.pi*np.arange(1, gf_dft.shape[-1], 2)/n_tau
    expm1 = np.expm1(d_tau_iws)
    weight1 = expm1/d_tau_iws
    weight2 = (expm1 + 1 - weight1)/d_tau_iws
    gf_iw = weight1*gf_dft[..., 1::2] + weight2*d_gf_dft[..., 1::2]
    gf_iw = -beta*gf_iw
    return gf_iw


def tau2iw(gf_tau, beta, n_pole=None, moments=None, fourier=tau2iw_ft_lin):
    r"""Fourier transform of the real Green's function `gf_tau`.

    Fourier transformation of a fermionic imaginary-time Green's function to
    Matsubara domain.
    We assume a real Green's function `gf_tau`, which is the case for
    commutator Green's functions :math:`G_{AB}(τ) = ⟨A(τ)B⟩` with
    :math:`A = B^†`. The Fourier transform `gf_iw` is then Hermitian.
    If no explicit `moments` are given, this function removes
    :math:`-G_{AB}(β) - G_{AB}(0) = ⟨[A,B]⟩`.

    Parameters
    ----------
    gf_tau : (..., N_tau) float np.ndarray
        The Green's function at imaginary times :math:`τ \in [0, β]`.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.
    n_pole : int, optional
        Number of poles used to fit `gf_tau`. Needs to be at least as large as
        the number of given moments `m`. (default: no fitting is performed)
    moments : (..., m) float array_like, optional
        High-frequency moments of `gf_iw`. If none are given, the first moment
        is chosen to remove the discontinuity at :math:`τ=0^{±}`.
    fourier : {`tau2iw_ft_lin`, `tau2iw_dft`}, optional
        Back-end to perform the actual Fourier transformation.

    Returns
    -------
    gf_iw : (..., {N_iv + 1}/2) complex np.ndarray
        The Fourier transform of `gf_tau` for non-negative fermionic Matsubara
        frequencies :math:`iω_n`.

    See Also
    --------
    tau2iw_ft_lin : Back-end: Fourier integration using Filon's method
    tau2iw_dft : Back-end: plain implementation using Riemann sum.

    pole_gf_from_tau : Function handling the fitting of `gf_tau`

    Examples
    --------
    >>> import gftool.fourier
    >>> BETA = 50
    >>> tau = np.linspace(0, BETA, num=2049, endpoint=True)
    >>> iws = gt.matsubara_frequencies(range((tau.size-1)//2), beta=BETA)

    >>> poles = 2*np.random.random(10) - 1  # partially filled
    >>> weights = np.random.random(10)
    >>> weights = weights/np.sum(weights)
    >>> gf_tau = gt.pole_gf_tau(tau, poles=poles, weights=weights, beta=BETA)
    >>> gf_ft = gt.fourier.tau2iw(gf_tau, beta=BETA)
    >>> gf_tau.size, gf_ft.size
    (2049, 1024)
    >>> gf_iw = gt.pole_gf_z(iws, poles=poles, weights=weights)

    >>> import matplotlib.pyplot as plt
    >>> __ = plt.plot(gf_iw.imag, label='exact Im')
    >>> __ = plt.plot(gf_ft.imag, '--', label='DFT Im')
    >>> __ = plt.plot(gf_iw.real, label='exact Re')
    >>> __ = plt.plot(gf_ft.real, '--', label='DFT Re')
    >>> __ = plt.legend()
    >>> plt.show()

    Accuracy of the different back-ends

    >>> ft_lin, dft = gt.fourier.tau2iw_ft_lin, gt.fourier.tau2iw_dft
    >>> gf_ft_lin = gt.fourier.tau2iw(gf_tau, beta=BETA, fourier=ft_lin)
    >>> gf_dft = gt.fourier.tau2iw(gf_tau, beta=BETA, fourier=dft)
    >>> __ = plt.plot(abs(gf_iw - gf_ft_lin), label='FT_lin')
    >>> __ = plt.plot(abs(gf_iw - gf_dft), '--', label='DFT')
    >>> __ = plt.legend()
    >>> plt.yscale('log')
    >>> plt.show()

    The accuracy can be further improved by fitting as suitable pole Green's
    function:

    >>> for n, n_mom in enumerate(range(1, 30, 5)):
    ...     gf = gt.fourier.tau2iw(gf_tau, n_pole=n_mom, moments=(1,), beta=BETA, fourier=ft_lin)
    ...     __ = plt.plot(abs(gf_iw - gf), label=f'n_fit={n_mom}', color=f'C{n}')
    ...     gf = gt.fourier.tau2iw(gf_tau, n_pole=n_mom, moments=(1,), beta=BETA, fourier=dft)
    ...     __ = plt.plot(abs(gf_iw - gf), '--', color=f'C{n}')
    >>> __ = plt.legend(loc='lower right')
    >>> plt.yscale('log')
    >>> plt.show()

    Results for DFT can be drastically improved giving high-frequency moments.
    The reason is, that lower large frequencies, where FT_lin is superior, are
    treated by the moments instead of the Fourier transform.

    >>> mom = np.sum(weights[:, np.newaxis] * poles[:, np.newaxis]**range(8), axis=0)
    >>> for n in range(1, 8):
    ...     gf = gt.fourier.tau2iw(gf_tau, moments=mom[:n], beta=BETA, fourier=ft_lin)
    ...     __ = plt.plot(abs(gf_iw - gf), label=f'n_mom={n}', color=f'C{n}')
    ...     gf = gt.fourier.tau2iw(gf_tau, moments=mom[:n], beta=BETA, fourier=dft)
    ...     __ = plt.plot(abs(gf_iw - gf), '--', color=f'C{n}')
    >>> __ = plt.legend(loc='lower right')
    >>> plt.yscale('log')
    >>> plt.show()

    The method is resistant against noise:

    >>> magnitude = 2e-7
    >>> noise = np.random.normal(scale=magnitude, size=gf_tau.size)
    >>> __, axes = plt.subplots(ncols=2, sharey=True)
    >>> for n, n_mom in enumerate(range(1, 20, 5)):
    ...     gf = gt.fourier.tau2iw(gf_tau + noise, n_pole=n_mom, moments=(1,), beta=BETA, fourier=ft_lin)
    ...     __ = axes[0].plot(abs(gf_iw - gf), label=f'n_fit={n_mom}', color=f'C{n}')
    ...     gf = gt.fourier.tau2iw(gf_tau + noise, n_pole=n_mom, moments=(1,), beta=BETA, fourier=dft)
    ...     __ = axes[1].plot(abs(gf_iw - gf), '--', color=f'C{n}')
    >>> for ax in axes:
    ...     __ = ax.axhline(magnitude, color='black')
    >>> __ = axes[0].legend()
    >>> plt.yscale('log')
    >>> plt.tight_layout()
    >>> plt.show()

    >>> __, axes = plt.subplots(ncols=2, sharey=True)
    >>> for n in range(1, 7, 2):
    ...     gf = gt.fourier.tau2iw(gf_tau + noise, moments=mom[:n], beta=BETA, fourier=ft_lin)
    ...     __ = axes[0].plot(abs(gf_iw - gf), '--', label=f'n_mom={n}', color=f'C{n}')
    ...     gf = gt.fourier.tau2iw(gf_tau + noise, moments=mom[:n], beta=BETA, fourier=dft)
    ...     __ = axes[1].plot(abs(gf_iw - gf), '--', color=f'C{n}')
    >>> for ax in axes:
    ...     __ = ax.axhline(magnitude, color='black')
    >>> __ = axes[0].plot(abs(gf_iw - gf_ft_lin), label='clean')
    >>> __ = axes[1].plot(abs(gf_iw - gf_dft), '--', label='clean')
    >>> __ = axes[0].legend(loc='lower right')
    >>> plt.yscale('log')
    >>> plt.tight_layout()
    >>> plt.show()

    """
    tau = np.linspace(0, beta, num=gf_tau.shape[-1])
    m1 = -gf_tau[..., -1] - gf_tau[..., 0]
    if moments is None:  # = 1/z moment = jump of Gf at 0^{±}
        moments = m1[..., newaxis]
    else:
        moments = np.asanyarray(moments)
        if not np.allclose(m1, moments[..., 0]):
            LOGGER.warning("Provided 1/z moment differs from jump."
                           "\n mom: %s, jump: %s", moments[..., 0], m1)
    if n_pole is None:
        n_pole = moments.shape[-1]
    # add additional axis for tau/iws for easy gu-function calling
    pole_gf = PoleGf.from_tau(gf_tau[..., newaxis, :], n_pole=n_pole, beta=beta,
                              moments=moments[..., newaxis, :])
    gf_tau = gf_tau - pole_gf.eval_tau(tau, beta)
    gf_iw = fourier(gf_tau, beta=beta)
    iws = gt.matsubara_frequencies(range(gf_iw.shape[-1]), beta=beta)
    gf_iw += pole_gf.eval_z(iws)
    return gf_iw
