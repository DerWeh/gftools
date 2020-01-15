"""Fourier transformations of Green's functions.

Fourier transformation between imaginary time and Matsubara frequencies.
The function in this module should be used after explicitly treating the
high-frequency behavior.


Glossary
--------

.. glossary::

   dft
      <discrete Foruier transform>

   ft
      <Fourier transformation> In contrast to :term:`dft`, this is used for
      Fourier integration of continous variables without discretization.

   iv
      <iν_n> Bosonic Matsubara frequncies

   iw
      <iω_n> Fermionic Matusbara frequncies

   tau
      Imaginary time points

"""
import logging

import numpy as np

import gftools as gt

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
    bandwidht!

    Examples
    --------
    >>> import gftools.fourier
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
    gf_tau = gf_tau[..., :gf_iwall.shape[-1]]  # trim to tau in [0, beta]
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
    iw2tau_dft : Plain implementation of fourier transform

    Notes
    -----
    For accurate an accurate Fourier transform, it is necessary, that `gf_iw`
    has already reached it's high-frequency behaviour, which need to be included
    explicitly. Therefore, the accuracy of the FT depends implicitely on the
    bandwidht!

    Examples
    --------
    >>> import gftools.fourier
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
    >>> import gftools.fourier
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
    gf_iv = beta * np.fft.ihfft(gf_tau[..., :-1] - gf_mean[..., np.newaxis])
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
    >>> import gftools.fourier
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
    fourier : {'tau2iv_ft_lin', 'tau2iv_dft'}, optional
        Back-end to perform the actual Fourier transformation.

    Returns
    -------
    gf_iv : (..., {N_iv + 1}/2) float np.ndarray
        The Fourier transform of `gf_tau` for non-negative bosonic Matsubara
        frequencies :math:`iν_n`.

    See Also
    --------
    tau2iv_dft : Back-end: plain implementation using Riemann sum.
    tau2iv_ft_lin : Back-end: plain implementation using Riemann sum.

    Examples
    --------
    >>> import gftools.fourier
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
    g1 = gf_tau[..., -1] - gf_tau[..., 0]  # = 1/z moment = jump of Gf at 0^{±}
    tau = np.linspace(0, beta, num=gf_tau.shape[-1])
    gf_tau = gf_tau - g1/beta*tau  # remove jump by linear shift
    gf_iv = fourier(gf_tau, beta=beta)
    ivs = gt.matsubara_frequencies_b(range(1, gf_iv.shape[-1]), beta=beta)
    gf_iv[1:] += g1/ivs
    gf_iv[0] += .5* g1 * beta  # `iv_{n=0}` = 0 has to be treated separately
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
    >>> import gftools.fourier
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

    Fourier transformation of a fermionc imaginary-time Green's function to
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
    >>> import gftools.fourier
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
    n_tau = gf_tau_full_range.shape[-1]
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
