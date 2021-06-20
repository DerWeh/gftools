"""Linear prediction to extrapolated retarded Green's function.

A nice introductory description of linear prediction is found in [Vaidyanathan2007]_.

References
----------
.. [Vaidyanathan2007] Vaidyanathan, P.P., 2007. The Theory of Linear Prediction.
   Synthesis Lectures on Signal Processing 2, 1–184.
   https://doi.org/10.2200/S00086ED1V01Y200712SPR003

"""
from functools import partial

import numpy as np


_gusum = partial(np.sum, axis=-1)


def pcoeff_burg(x, order: int):
    """Burg's method for linear prediction (LP) coefficients.

    Burg's method calculates the coefficients from an estimate of the reflection
    coefficients using Levinson's method. Burg's method guarantees that poles
    are inside the unit circle [Kay1988]_, thus it is stable.

    Parameters
    ----------
    x : (..., N) complex np.ndarray
        Data of the (time) series to be predicted.
    order : int
        Prediction order, has to be smaller then `N`.

    Returns
    -------
    a : (..., order) complex np.ndarray
        Prediction coefficients
    rho : (...) float np.ndarray
        Variance estimate.

    References
    ----------
    .. [Kay1988] Kay, S.M., 1988. Modern spectral estimation:
       theory and application. Pearson Education India.

    """
    N = x.shape[-1]
    rho = np.full([*x.shape[:-1], order+1], np.nan)
    rho[..., 0] = _gusum(abs(x)**2) / N
    eforw = np.ascontiguousarray(x[..., 1:])
    eback = np.ascontiguousarray(x[..., :-1])
    a = np.full([*x.shape[:-1], order, order], np.nan, dtype=x.dtype)
    for k in range(1, order+1):
        # eq (7.38)
        numer = -2*_gusum(eforw*np.conj(eback))
        # eq (7.42) for more efficient denom
        # if k > 1:
        #     denom = (1.0 -  np.abs(aa)**2)*denom - np.abs(eforw)
        # else:
        denom = _gusum(abs(eforw)**2) + _gusum(abs(eback)**2)
        # eq (7.40)
        kk = a[..., k-1, k-1] = numer / denom
        rho[..., k] = (1.0 - abs(kk)**2) * rho[..., k-1]
        kk = kk[..., np.newaxis]
        if k > 1:
            a[..., k-1, :k-1] = a[..., k-2, :k-1] + kk*np.conj(a[..., k-2, k-2::-1])
        # eq (7.41)
        eforw, eback = (eforw[..., 1:] + kk*eback[..., 1:],
                        eback[..., :-1] + np.conj(kk)*eforw[..., :-1])
    sig2 = rho[..., -1]
    a = np.ascontiguousarray(a[..., -1, :])
    return a, sig2


def predict(x, pcoeff, num: int):
    """Forward-predict a series additional `num` steps.

    Parameters
    ----------
    x : (..., N) complex np.ndarray
        Data of the (time) series to be predicted.
    pcoeff : (..., order) complex np.ndarray
        Prediction coefficients
    num : int
        Number of additional (time) steps.

    Returns
    -------
    px : (..., N+num) complex np.ndarray
        Data of the (time) series extended by `num` steps, with
        `px[:x.size] = x`.

    """
    start = x.shape[-1]
    order = pcoeff.shape[-1]
    xtended = np.concatenate([x, np.full([*x.shape[:-1], num], np.nan, dtype=x.dtype)], axis=-1)
    for ii in range(start, start+num):
        xtended[..., ii] = -_gusum(pcoeff[..., ::-1] * xtended[..., ii-order:ii])
    return xtended
