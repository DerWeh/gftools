"""Fourier transformations of Green's functions.

Glossary
--------

.. glossary::

   dft
      discrete Foruier transform

   iv
      Bosonic Matsubara frequncies

   iw
      <iω_n> Fermionic Matusbara frequncies

   tau
      Imaginary time points

"""
import numpy as np


def tau2iw_dft(gf_tau, beta):
    r"""Discrete Fourier transfrom of the real Green's function `gf_tau`.

    Fourier transformation of a fermionc imaginary-time Green's function to
    Matsubara domain.
    The fourier integral is replaced by a Riemann sum giving a discrete
    Fourier transform (DFT).
    We assume a real Green's function `gf_tau`, which is the case for
    commutator Green's functions :math:`G_{AB}(τ) = ⟨A(τ)B⟩` with
    :math:`A = B^†`.

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

    The method is resistent against noise:

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

    Filon's method is used to calculated the Fourier integral

    .. math:: G^n = .5 ∫_{-β}^{β}dτ G(τ) e^{iω_n τ},

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

    Notes
    -----
    This function performs better than `bare_dft_tau2iw` for high frequencies,
    for low frequencies `bare_dft_tau2iw` might perform better.
    To my experience, `lin_ft_tau2iw` can be used for all frequencies if the
    first two moments are stripped using `dft_tau2iw`.

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

    The method is resistent against noise:

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
