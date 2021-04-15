r"""Fourier transformations of Green's functions.

Fourier transformation between imaginary time and Matsubara frequencies.
The function in this module should be used after explicitly treating the
high-frequency behavior, as this is not yet implemented.
Typically, transformation from τ-space to Matsubara frequency are unproblematic.

The Fourier transforms are defined in the following way:

Definitions
-----------

real time → complex frequencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Laplace integral for the Green's function is defined as

.. math:: G(z) = ∫_{-∞}^{∞} dt G(t) \exp(izt)

This integral is only well defined

* in the upper complex half-plane `z.imag>=0` for retarded Green's function :math:`∝θ(t)`
* in the lower complex half-plane `z.imag<=0` for advanced Green's function :math:`∝θ(-t)`

The recommended high-level function to perform this Laplace transform is:

* `tt2z` for both retarded and advanced Green's function

Currently, to sub-functions can be used equivalently, the abstraction `tt2z` is
mostly for consistency with the imaginary time ↔ Matsubara frequencies
Fourier transformations.

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

from gftool._util import _gu_matvec
from gftool.statistics import matsubara_frequencies, matsubara_frequencies_b
from gftool.basis.pole import PoleFct, PoleGf

try:
    import numexpr as ne
except ImportError:
    _HAS_NUMEXPR = False
else:
    _HAS_NUMEXPR = True


LOGGER = logging.getLogger(__name__)


def _phase_numexpr(z, tt):
    return ne.evaluate('exp(1j*z*tt)', local_dict={'z': z, 'tt': tt})


def _phase_numpy(z, tt):
    return np.exp(1j*z*tt)


_phase = _phase_numexpr if _HAS_NUMEXPR else _phase_numpy


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
    gf_iw : (..., N_iw) complex np.ndarray
        The Green's function at positive **fermionic** Matsubara frequencies
        :math:`iω_n`.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.
    moments : (..., m) float array_like
        High-frequency moments of `gf_iw`.
    fourier : {`iw2tau_dft`, `iw2tau_dft_soft`}, optional
        Back-end to perform the actual Fourier transformation.
    n_fit : int, optional
        Number of additionally fitted moments (in fact, `gf_iw` is fitted, not
        not directly moments).

    Returns
    -------
    gf_tau : (..., 2*N_iw + 1) float np.ndarray
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
    iws = matsubara_frequencies(range(gf_iw.shape[-1]), beta=beta)
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
    gf_iv : (..., (N_iv + 1)/2) float np.ndarray
        The Fourier transform of `gf_tau` for non-negative bosonic Matsubara
        frequencies :math:`iν_n`.

    See Also
    --------
    tau2iv_ft_lin : Fourier integration using Filon's method

    Examples
    --------
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
    gf_iv : (..., (N_iv + 1)/2) float np.ndarray
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
    d_tau_ivs[..., 0] = 1  # avoid zero division, fix value by hand later
    expm1 = np.expm1(d_tau_ivs)
    weight1 = expm1/d_tau_ivs
    weight2 = (expm1 + 1 - weight1)/d_tau_ivs
    weight1[..., 0], weight2[..., 0] = 1, .5  # special case n=0, fix from before
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
    gf_iv : (..., (N_iv + 1)/2) complex np.ndarray
        The Fourier transform of `gf_tau` for non-negative bosonic Matsubara
        frequencies :math:`iν_n`.

    See Also
    --------
    tau2iv_dft : Back-end: plain implementation using Riemann sum.
    tau2iv_ft_lin : Back-end: Fourier integration using Filon's method.

    Examples
    --------
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
    ivs = matsubara_frequencies_b(range(1, gf_iv.shape[-1]), beta=beta)
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
    gf_iw : (..., (N_iw - 1)/2) float np.ndarray
        The Fourier transform of `gf_tau` for positive fermionic Matsubara
        frequencies :math:`iω_n`.

    See Also
    --------
    tau2iw_ft_lin : Fourier integration using Filon's method

    Examples
    --------
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
    gf_iw : (..., (N_iw - 1)/2) float np.ndarray
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
    gf_iw : (..., (N_iv + 1)/2) complex np.ndarray
        The Fourier transform of `gf_tau` for non-negative fermionic Matsubara
        frequencies :math:`iω_n`.

    See Also
    --------
    tau2iw_ft_lin : Back-end: Fourier integration using Filon's method
    tau2iw_dft : Back-end: plain implementation using Riemann sum.

    pole_gf_from_tau : Function handling the fitting of `gf_tau`

    Examples
    --------
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
    iws = matsubara_frequencies(range(gf_iw.shape[-1]), beta=beta)
    gf_iw += pole_gf.eval_z(iws)
    return gf_iw


def _z2polegf(z, gf_z, n_pole, moments=(1.,)) -> PoleFct:
    moments = np.asanyarray(moments)

    def error_(width):
        pole_gf = PoleFct.from_z(z, gf_z, n_pole=n_pole,
                                 # if width is 0, no higher moments exist
                                 moments=moments if width else moments[..., 0:1], width=width)
        gf_fit = pole_gf.eval_z(z)
        return np.linalg.norm(gf_z - gf_fit)

    from scipy.optimize import minimize_scalar  # pylint: disable=import-outside-toplevel
    opt = minimize_scalar(error_)
    LOGGER.debug("Fitting error: %s Optimal pole-spread: %s", opt.fun, opt.x)
    opt_pole_gf = PoleFct.from_z(z, gf_z, n_pole=n_pole, moments=moments, width=opt.x)
    return opt_pole_gf


def izp2tau(izp, gf_izp, tau, beta, moments=(1.,)):
    r"""Fourier transform of the Hermitian Green's function `gf_izp` to `tau`.

    Fourier transformation of a fermionic Padé Green's function to
    imaginary-time domain.
    We assume a Hermitian Green's function `gf_iw`, i.e. :math:`G(-iω_n) = G^*(iω_n)`,
    which is the case for commutator Green's functions :math:`G_{AB}(τ) = ⟨A(τ)B⟩`
    with :math:`A = B^†`. The Fourier transform `gf_tau` is then real.

    TODO: this function is not vectorized yet.

    Parameters
    ----------
    izp, gf_iw : (N_izp) float np.ndarray
        Positive **fermionic** Padé frequencies :math:`iz_p` and the Green's
        function at specified frequencies.
    tau : (N_tau) float np.ndarray
        Imaginary times `0 <= tau <= beta` at which the Fourier transform is
        evaluated.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.
    moments : (m) float array_like, optional
        High-frequency moments of `gf_iw`.

    Returns
    -------
    gf_tau : (N_tau) float np.ndarray
        The Fourier transform of `gf_izp` for imaginary times `tau`.

    See Also
    --------
    iw2tau : Fourier transform from fermionic Matsubara frequencies.
    _z2polegf : Function handling the fitting of `gf_izp`

    Notes
    -----
    The algorithm performs in fact an analytic continuation instead of a
    Fourier integral. It is however only evaluated on the imaginary axis, so
    far the algorithm was observed to be stable

    Examples
    --------
    >>> BETA = 50
    >>> izp, __ = gt.pade_frequencies(50, beta=BETA)
    >>> tau = np.linspace(0, BETA, num=2049, endpoint=True)

    >>> poles = 2*np.random.random(10) - 1  # partially filled
    >>> weights = np.random.random(10)
    >>> weights = weights/np.sum(weights)
    >>> gf_izp = gt.pole_gf_z(izp, poles=poles, weights=weights)
    >>> gf_ft = gt.fourier.izp2tau(izp, gf_izp, tau, beta=BETA)
    >>> gf_tau = gt.pole_gf_tau(tau, poles=poles, weights=weights, beta=BETA)

    >>> import matplotlib.pyplot as plt
    >>> __ = plt.plot(tau, gf_tau, label='exact')
    >>> __ = plt.plot(tau, gf_ft, '--', label='FT')
    >>> __ = plt.legend()
    >>> plt.show()

    >>> __ = plt.title('Oscillations of tiny magnitude')
    >>> __ = plt.plot(tau/BETA, gf_tau - gf_ft)
    >>> __ = plt.xlabel('τ/β')
    >>> plt.show()

    Results of `izp2tau` can be improved giving high-frequency moments.

    >>> mom = np.sum(weights[:, np.newaxis] * poles[:, np.newaxis]**range(4), axis=0)
    >>> for n in range(1, 4):
    ...     gf = gt.fourier.izp2tau(izp, gf_izp, tau, beta=BETA, moments=mom[:n])
    ...     __ = plt.plot(tau, abs(gf_tau - gf), label=f'n_mom={n}')
    >>> __ = plt.legend()
    >>> plt.yscale('log')
    >>> plt.show()

    The method is resistant against noise:

    >>> magnitude = 2e-7
    >>> noise = np.random.normal(scale=magnitude, size=gf_izp.size)
    >>> gf = gt.fourier.izp2tau(izp, gf_izp + noise, tau, beta=BETA, moments=(1,))
    >>> __ = plt.plot(tau/BETA, abs(gf_tau - gf))
    >>> __ = plt.axhline(magnitude, color='black')
    >>> plt.yscale('log')
    >>> plt.tight_layout()
    >>> plt.show()

    >>> for n in range(1, 4):
    ...     gf = gt.fourier.izp2tau(izp, gf_izp + noise, tau, beta=BETA, moments=mom[:n])
    ...     __ = plt.plot(tau/BETA, abs(gf_tau - gf), '--', label=f'n_mom={n}')
    >>> __ = plt.axhline(magnitude, color='black')
    >>> __ = plt.plot(tau/BETA, abs(gf_tau - gf_ft), label='clean')
    >>> __ = plt.legend()
    >>> plt.yscale('log')
    >>> plt.tight_layout()
    >>> plt.show()

    """
    pole_gf = PoleGf(*_z2polegf(izp, gf_izp, n_pole=izp.size, moments=moments))
    return pole_gf.eval_tau(tau, beta)


def tt2z_trapz(tt, gf_t, z):
    r"""Laplace transform of the real-time Green's function `gf_t`.

    Approximate the Laplace integral by trapezoidal rule:

    .. math::

       G(z) = ∫dt G(t) \exp(izt)
            ≈ ∑_{k=1}^N [G(t_{k-1})\exp(izt_{k-1}) + G(t_k)\exp(izt_k)] Δt_k/2

    The function can handle any input discretization `tt`.

    Parameters
    ----------
    tt : (Nt) float np.ndarray
        The points for which the Green's function `gf_t` is given.
    gf_t : (..., Nt) complex np.ndarray
        Green's function and time points `tt`.
    z : (..., Nz) complex np.ndarray
        Frequency points for which the Laplace transformed Green's function
        should be evaluated.

    Returns
    -------
    gf_z : (..., Nz) complex np.ndarray
        Laplace transformed Green's function for complex frequencies `z`.

    See Also
    --------
    tt2z_lin : Laplace integration using Filon's method

    Notes
    -----
    The function is equivalent to the one-liner
    `np.trapz(np.exp(1j*z[:, None]*tt)*gf_t, x=tt)`.
    Internally this function evaluates the sum as a matrix product to leverage
    the speed-up of BLAS. If `numexpr` is available, it is used for the speed
    up it provides for transcendental equations.

    """
    phase = _phase(z[..., newaxis], tt[newaxis, :])
    boundary = (phase[..., 0]*gf_t[..., :1]*(tt[1] - tt[0])
                + phase[..., -1]*gf_t[..., -1:]*(tt[-1] - tt[-2]))
    d2tt = tt[2:] - tt[:-2]
    trapz = _gu_matvec(phase[..., 1:-1], gf_t[..., 1:-1]*d2tt)
    return 0.5*(boundary + trapz)


def tt2z_lin(tt, gf_t, z):
    r"""Laplace transform of the real-time Green's function `gf_t`.

    Filon's method is used to calculate the Laplace integral

    .. math:: G(z) = ∫dt G(t) \exp(izt),

    :math:`G(t)` is approximated by a linear spline.
    The function currently requires an equidistant `tt`.
    Information on oscillatory integrations can be found e.g. in [filon1928]_
    and [iserles]_.

    Parameters
    ----------
    tt : (Nt) float np.ndarray
        The equidistant points for which the Green's function `gf_t` is given.
    gf_t : (..., Nt) complex np.ndarray
        Green's function and time points `tt`.
    z : (..., Nz) complex np.ndarray
        Frequency points for which the Laplace transformed Green's function
        should be evaluated.

    Returns
    -------
    gf_z : (..., Nz) complex np.ndarray
        Laplace transformed Green's function for complex frequencies `z`.

    Raises
    ------
    ValueError
        If the time points `tt` are not equidistant.

    See Also
    --------
    tt2z_trapz : Plain implementation using trapezoidal rule.

    Notes
    -----
    Internally this function evaluates the sum as a matrix product to leverage
    the speed-up of BLAS. If `numexpr` is available, it is used for the speed
    up it provides for transcendental equations.

    References
    ----------
    .. [filon1928] L.N. Filon, On a quadrature formula for trigonometric integrals,
       Proc. Roy. Soc. Edinburgh 49 (1928) 38-47.
    .. [iserles] A. Iserles, S.P. Nørsett, and S. Olver, Highly oscillatory
       quadrature: The story so far,
       http://www.sam.math.ethz.ch/~hiptmair/Seminars/OSCINT/INO06.pdf

    """
    delta_tt = tt[1] - tt[0]
    if not np.allclose(tt[1:] - tt[:-1], delta_tt):
        raise ValueError("Equidistant `tt` required for current implementation.")
    zero = z == 0  # special case `z=0` has to be handled separately (due: 1/z)
    if np.any(zero):
        z = np.where(zero, 1, z)
    izdt = 1j*z*delta_tt
    phase = _phase(z[..., newaxis], tt[newaxis, :-1])
    g_dft = _gu_matvec(phase, gf_t[..., :-1])
    dg_dft = _gu_matvec(phase, gf_t[..., 1:] - gf_t[..., :-1])
    weight1 = np.expm1(izdt)/izdt
    weight2 = (np.exp(izdt) - weight1)/izdt
    gf_z = delta_tt * (weight1*g_dft + weight2*dg_dft)
    if np.any(zero):
        gf_z[..., zero] = np.trapz(gf_t, x=tt)[..., np.newaxis]
    return gf_z


def tt2z(tt, gf_t, z, laplace=tt2z_lin):
    r"""Laplace transform of the real-time Green's function `gf_t`.

    Calculate the Laplace transform

    .. math:: G(z) = ∫dt G(t) \exp(izt)

    For the Laplace transform to be well defined,
    it should either be `tt>=0 and z.imag>=0` for the retarded Green's function,
    or `tt<=0 and z.imag<=0` for the advance Green's function.

    The retarded (advanced) Green's function can in principle be evaluated for
    any frequency point `z` in the upper (lower) complex half-plane.

    The accented contours for `tt` and `z` depend on the specific used back-end
    `laplace`.

    Parameters
    ----------
    tt : (Nt) float np.ndarray
        The points for which the Green's function `gf_t` is given.
    gf_t : (..., Nt) complex np.ndarray
        Green's function and time points `tt`.
    z : (..., Nz) complex np.ndarray
        Frequency points for which the Laplace transformed Green's function
        should be evaluated.
    laplace : {`tt2z_lin`, `tt2z_trapz`}, optional
        Back-end to perform the actual Fourier transformation.

    Returns
    -------
    gf_z : (..., Nz) complex np.ndarray
        Laplace transformed Green's function for complex frequencies `z`.

    See Also
    --------
    tt2z_trapz : Back-end: approximate integral by trapezoidal rule
    tt2z_lin : Back-end: approximate integral by Filon's method

    Raises
    ------
    ValueError
        If neither the condition for retarded or advanced Green's function is
        fulfilled.

    Examples
    --------
    >>> tt = np.linspace(0, 150, num=1501)
    >>> ww = np.linspace(-2, 2, num=501) + 1e-1j

    >>> poles = 2*np.random.random(10) - 1  # partially filled
    >>> weights = np.random.random(10)
    >>> weights = weights/np.sum(weights)
    >>> gf_ret_t = gt.pole_gf_ret_t(tt, poles=poles, weights=weights)
    >>> gf_ft = gt.fourier.tt2z(tt, gf_ret_t, z=ww)
    >>> gf_ww = gt.pole_gf_z(ww, poles=poles, weights=weights)

    >>> import matplotlib.pyplot as plt
    >>> __ = plt.plot(ww.real, gf_ww.imag, label='exact Im')
    >>> __ = plt.plot(ww.real, gf_ft.imag, '--', label='DFT Im')
    >>> __ = plt.plot(ww.real, gf_ww.real, label='exact Re')
    >>> __ = plt.plot(ww.real, gf_ft.real, '--', label='DFT Re')
    >>> __ = plt.legend()
    >>> plt.show()

    The function Laplace transform can be evaluated at abitrary contours,
    e.g. for a semi-ceircle in the the upper half-plane.
    Note, that close to the real axis the accuracy is bad, due to the
    truncation at `max(tt)`

    >>> z = np.exp(1j*np.pi*np.linspace(0, 1, num=51))
    >>> gf_ft = gt.fourier.tt2z(tt, gf_ret_t, z=z)
    >>> gf_z = gt.pole_gf_z(z, poles=poles, weights=weights)

    >>> import matplotlib.pyplot as plt
    >>> __ = plt.plot(z.real, gf_z.imag, '+', label='exact Im')
    >>> __ = plt.plot(z.real, gf_ft.imag, 'x', label='DFT Im')
    >>> __ = plt.plot(z.real, gf_z.real, '+', label='exact Re')
    >>> __ = plt.plot(z.real, gf_ft.real, 'x', label='DFT Re')
    >>> __ = plt.legend()
    >>> plt.show()

    Accuracy of the different back-ends:

     * For large `z.imag`, `tt2z_lin` performs better.
     * For intermediate `z.imag`, the quality depends on the relevant `z.real`.
       For small `z.real`, the error of `tt2z_trapz` is more uniform;
       for big `z.real`, `tt2z_lin` is a good approximation.
     * For small `z.imag`, the methods are almost identical,
       the truncation of `tt` dominates the error.

    >>> import matplotlib.pyplot as plt
    >>> for ii, eta in enumerate([1.0, 0.5, 0.1, 0.03]):
    ...     ww.imag = eta
    ...     gf_ww = gt.pole_gf_z(ww, poles=poles, weights=weights)
    ...     gf_trapz = gt.fourier.tt2z(tt, gf_ret_t, z=ww, laplace=gt.fourier.tt2z_trapz)
    ...     gf_lin = gt.fourier.tt2z(tt, gf_ret_t, z=ww, laplace=gt.fourier.tt2z_lin)
    ...     __ = plt.plot(ww.real, abs((gf_ww - gf_trapz)/gf_ww),
    ...                   label=f"z.imag={eta}", color=f"C{ii}")
    ...     __ = plt.plot(ww.real, abs((gf_ww - gf_lin)/gf_ww), '--', color=f"C{ii}")
    ...     __ = plt.legend()
    >>> plt.yscale('log')
    >>> plt.show()

    """
    retarded = np.all(tt >= 0) and np.all(z.imag >= 0)
    advanced = np.all(tt <= 0) and np.all(z.imag <= 0)
    if not (retarded or advanced):
        raise ValueError("Laplace Transform only well defined if `tt>=0 and z.imag>=0`"
                         " or `tt<=0 and z.imag<=0`")
    if z.size == 0:  # consistent behavior for gufuncs
        return np.empty_like(z)
    return laplace(tt, gf_t, z)


def _tau2polegf(gf_tau, beta, n_pole, moments=None, occ=False, weight=None) -> PoleGf:
    tau = np.linspace(0, beta, num=gf_tau.shape[-1])
    m1 = -gf_tau[..., -1] - gf_tau[..., 0]
    if moments is None:  # = 1/z moment = jump of Gf at 0^{±}
        moments = m1[..., newaxis]
    else:
        moments = np.asanyarray(moments)
        if not np.allclose(m1, moments[..., 0]):
            LOGGER.warning("Provided 1/z moment differs from jump."
                           "\n mom: %s, jump: %s", moments[..., 0], m1)

    def error_(width):
        pole_gf = PoleGf.from_tau(gf_tau, n_pole=n_pole, beta=beta,
                                  # if width is 0, no higher moments exist
                                  moments=moments if width else m1[..., newaxis],
                                  occ=occ, width=width, weight=weight)
        gf_fit = pole_gf.eval_tau(tau, beta=beta)
        return np.linalg.norm(gf_tau - gf_fit)

    from scipy.optimize import minimize_scalar  # pylint: disable=import-outside-toplevel
    opt = minimize_scalar(error_)
    LOGGER.debug("Fitting error: %s Optimal pole-spread: %s", opt.fun, opt.x)
    opt_pole_gf = PoleGf.from_tau(gf_tau, n_pole=n_pole, beta=beta, moments=moments,
                                  occ=occ, width=opt.x, weight=weight)
    return opt_pole_gf


def tau2izp(gf_tau, beta, izp, moments=None, occ=False, weight=None):
    r"""Fourier transform of the real Green's function `gf_tau` to `izp`.

    Fourier transformation of a fermionic imaginary-time Green's function to
    fermionic imaginary Padé frequencies `izp`.
    We assume a real Green's function `gf_tau`, which is the case for
    commutator Green's functions :math:`G_{AB}(τ) = ⟨A(τ)B⟩` with
    :math:`A = B^†`. The Fourier transform `gf_iw` is then Hermitian.
    If no explicit `moments` are given, this function removes
    :math:`-G_{AB}(β) - G_{AB}(0) = ⟨[A,B]⟩`.

    TODO: this function is not vectorized yet.

    Parameters
    ----------
    gf_tau : (N_tau) float np.ndarray
        The Green's function at imaginary times :math:`τ \in [0, β]`.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.
    izp : (N_izp) complex np.ndarray
        Complex Padé frequencies at which the Fourier transform is evaluated.
    moments : (m) float array_like, optional
        High-frequency moments of `gf_iw`. If none are given, the first moment
        is chosen to remove the discontinuity at :math:`τ=0^{±}`.

    Returns
    -------
    gf_izp : (N_izp) complex np.ndarray
        The Fourier transform of `gf_tau` for given Padé frequencies `izp`.

    See Also
    --------
    tau2iw : Fourier transform to fermionic Matsubara frequencies.

    pole_gf_from_tau : Function handling the fitting of `gf_tau`

    Notes
    -----
    The algorithm performs in fact an analytic continuation instead of a
    Fourier integral. It is however only evaluated on the imaginary axis, so
    far the algorithm was observed to be stable

    Examples
    --------
    >>> BETA = 50
    >>> tau = np.linspace(0, BETA, num=2049, endpoint=True)
    >>> izp, __ = gt.pade_frequencies(50, beta=BETA)

    >>> poles = 2*np.random.random(10) - 1  # partially filled
    >>> weights = np.random.random(10)
    >>> weights = weights/np.sum(weights)
    >>> gf_tau = gt.pole_gf_tau(tau, poles=poles, weights=weights, beta=BETA)
    >>> gf_ft = gt.fourier.tau2izp(gf_tau, beta=BETA, izp=izp)
    >>> gf_izp = gt.pole_gf_z(izp, poles=poles, weights=weights)

    >>> import matplotlib.pyplot as plt
    >>> __ = plt.plot(gf_izp.imag, label='exact Im')
    >>> __ = plt.plot(gf_ft.imag, '--', label='FT Im')
    >>> __ = plt.plot(gf_izp.real, label='exact Re')
    >>> __ = plt.plot(gf_ft.real, '--', label='FT Re')
    >>> __ = plt.legend()
    >>> plt.show()

    Results of `tau2izp` can be improved giving high-frequency moments.

    >>> mom = np.sum(weights[:, np.newaxis] * poles[:, np.newaxis]**range(6), axis=0)
    >>> for n in range(1, 6):
    ...     gf = gt.fourier.tau2izp(gf_tau, izp=izp, moments=mom[:n], beta=BETA)
    ...     __ = plt.plot(abs(gf_izp - gf), label=f'n_mom={n}', color=f'C{n}')
    >>> __ = plt.legend()
    >>> plt.yscale('log')
    >>> plt.show()

    The method is resistant against noise,
    especially if there is knowledge of the noise:

    >>> magnitude = 2e-7
    >>> noise = np.random.normal(scale=magnitude, size=gf_tau.size)
    >>> gf = gt.fourier.tau2izp(gf_tau + noise, izp=izp, moments=(1,), beta=BETA)
    >>> __ = plt.plot(abs(gf_izp - gf), label='bare')
    >>> gf = gt.fourier.tau2izp(gf_tau + noise, izp=izp, moments=(1,), beta=BETA,
    ...                         weight=abs(noise)**-0.5)
    >>> __ = plt.plot(abs(gf_izp - gf), label='weighted')
    >>> __ = plt.axhline(magnitude, color='black')
    >>> __ = plt.legend()
    >>> plt.yscale('log')
    >>> plt.tight_layout()
    >>> plt.show()

    >>> for n in range(1, 7, 2):
    ...     gf = gt.fourier.tau2izp(gf_tau + noise, izp=izp, moments=mom[:n], beta=BETA)
    ...     __ = plt.plot(abs(gf_izp - gf), '--', label=f'n_mom={n}', color=f'C{n}')
    >>> __ = plt.axhline(magnitude, color='black')
    >>> __ = plt.plot(abs(gf_izp - gf_ft), label='clean')
    >>> __ = plt.legend()
    >>> plt.yscale('log')
    >>> plt.tight_layout()
    >>> plt.show()

    """
    pole_gf = _tau2polegf(gf_tau, beta, n_pole=izp.size, moments=moments, occ=occ, weight=weight)
    return pole_gf.eval_z(izp)
