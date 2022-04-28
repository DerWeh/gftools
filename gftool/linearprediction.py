"""Linear prediction to extrapolated retarded Green's function.

A nice introductory description of linear prediction can be found in [Vaidyanathan2007]_;
[Makhoul1975]_ gives a detailed review.

Linear prediction allows to extend a time series by predicting future points
:math:`x̂_l` as a linear combination of previous time points :math:`x`:

.. math:: x̂_l = -∑_{k=1}^{K} x_{l-k} a_k

where :math:`a_k` are the prediction coefficients and :math:`K` is the
prediction order.
**Recommended** method to obtain the prediction coefficients is `pcoeff_covar`,
the results of `pcoeff_burg` seem to be rather **unreliable**.

References
----------
.. [Vaidyanathan2007] Vaidyanathan, P.P., 2007. The Theory of Linear Prediction.
   Synthesis Lectures on Signal Processing 2, 1–184.
   https://doi.org/10.2200/S00086ED1V01Y200712SPR003
.. [Makhoul1975] Makhoul, J. Linear prediction: A tutorial review.
   Proceedings of the IEEE 63, 561–580 (1975).
   https://doi.org/10.1109/PROC.1975.9792

Examples
--------
We consider the retarded-time Bethe Green's function with the known time points

.. plot::
   :format: doctest
   :context: close-figs

   >>> tt = np.linspace(0, 10, 101)
   >>> gf_t = gt.lattice.bethe.gf_ret_t(tt, half_bandwidth=1)

We can predict future time points using linear prediction,
let's check the next 100 time points

.. plot::
   :format: doctest
   :context: close-figs

   >>> lp = gt.linearprediction
   >>> pcoeff, __ = lp.pcoeff_covar(gf_t, order=gf_t.size//2)
   >>> gf_pred = lp.predict(gf_t, pcoeff, num=100)

and compare the results

.. plot::
   :format: doctest
   :context: close-figs

   >>> import matplotlib.pyplot as plt
   >>> tt_pred = np.linspace(0, 20, 201)
   >>> gf_full = gt.lattice.bethe.gf_ret_t(tt_pred, half_bandwidth=1)
   >>> __ = plt.axhline(0, color='dimgray', linewidth=0.8)
   >>> __ = plt.axvline(tt[-1], color='dimgray', linewidth=0.8)
   >>> __ = plt.plot(tt_pred, gf_full.imag)
   >>> __ = plt.plot(tt_pred, gf_pred.imag, '--')
   >>> plt.show()

The roots corresponding to the linear prediction polynomial should all lie in
the unit circle, numerical inaccuracies can lead to roots outside the unit
circle causing exponentially growing contributions.
For example, if we add some noise:

.. plot::
   :format: doctest
   :context: close-figs

   >>> noise = np.random.default_rng(0).normal(scale=1e-6, size=tt.size)
   >>> pcoeff, __ = lp.pcoeff_covar(gf_t + noise, order=gf_t.size//2)
   >>> __ = lp.plot_roots(pcoeff)

The red crosses correspond to growing contributions. Prediction for long times
produces exponentially growing errors:

.. plot::
   :format: doctest
   :context: close-figs

   >>> import matplotlib.pyplot as plt
   >>> tt_pred = np.linspace(0, 30, 301)
   >>> gf_full = gt.lattice.bethe.gf_ret_t(tt_pred, half_bandwidth=1)
   >>> gf_pred = lp.predict(gf_t, pcoeff, num=200)
   >>> __ = plt.axhline(0, color='dimgray', linewidth=0.8)
   >>> __ = plt.axvline(tt[-1], color='dimgray', linewidth=0.8)
   >>> __ = plt.plot(tt_pred, gf_full.imag)
   >>> __ = plt.plot(tt_pred, gf_pred.imag, '--')
   >>> plt.show()

This can be amended by setting `stable=True` in `~gftool.linearprediction.predict`:

.. plot::
   :format: doctest
   :context: close-figs

   >>> import matplotlib.pyplot as plt
   >>> tt_pred = np.linspace(0, 30, 301)
   >>> gf_full = gt.lattice.bethe.gf_ret_t(tt_pred, half_bandwidth=1)
   >>> gf_pred = lp.predict(gf_t, pcoeff, num=200, stable=True)
   >>> __ = plt.axhline(0, color='dimgray', linewidth=0.8)
   >>> __ = plt.axvline(tt[-1], color='dimgray', linewidth=0.8)
   >>> __ = plt.plot(tt_pred, gf_full.imag)
   >>> __ = plt.plot(tt_pred, gf_pred.imag, '--')
   >>> plt.show()

"""
import numpy as np

from scipy.linalg import toeplitz

from gftool._util import _gu_sum
from gftool.matrix import decompose_mat


def companion(a):
    """Create a companion matrix.

    Create the companion matrix [1]_ associated with the polynomial whose
    coefficients are given in `a`.

    Parameters
    ----------
    a : (N,) array_like
        1-D array of polynomial coefficients. The length of `a` must be
        at least two, and ``a[0]`` must not be zero.

    Returns
    -------
    c : (N-1, N-1) ndarray
        The first row of `c` is ``-a[1:]/a[0]``, and the first
        sub-diagonal is all ones.  The data-type of the array is the same
        as the data-type of ``1.0*a[0]``.

    Raises
    ------
    ValueError
        If any of the following are true: a) ``a.ndim != 1``;
        b) ``a.size < 2``; c) ``a[0] == 0``.

    Notes
    -----
    Modified version of SciPy, contribute it back

    References
    ----------
    .. [1] R. A. Horn & C. R. Johnson, *Matrix Analysis*.  Cambridge, UK:
        Cambridge University Press, 1999, pp. 146-7.

    """
    a = np.atleast_1d(a)

    if a.size < 2:
        raise ValueError("The length of `a` must be at least 2.")

    if a[0] == 0:
        raise ValueError("The first coefficient in `a` must not be zero.")

    first_row = -a[1:] / (1.0 * a[0])
    n = a.shape[-1]

    c = np.zeros(a.shape[:-1] + (n - 1, n - 1), dtype=first_row.dtype)
    c[..., 0, :] = first_row
    c[..., list(range(1, n - 1)), list(range(0, n - 2))] = 1
    return c


def pcoeff_covar(x, order: int, rcond=None):
    """Calculate linear prediction (LP) coefficients using covariance method.

    The covariance method gives the equation

    .. math:: Ra = X^†X a = X^†x = -r

    where :math:`R` is the covariance matrix and :math:`a` are the LP
    coefficients.
    We solve :math:`Xa = x` using linear least-squares.

    Parameters
    ----------
    x : (..., N) complex np.ndarray
        Data of the (time) series to be predicted.
    order : int
        Prediction order, has to be smaller then `N`.
    rcond : float, optional
        Cut-off ratio for small singular values of `a`.
        For the purposes of rank determination, singular values are treated
        as zero if they are smaller than `rcond` times the largest singular
        value of `a`.

    Returns
    -------
    a : (..., order) complex np.ndarray
        Prediction coefficients.
    rho : (...) float np.ndarray
        Error estimate :math:`‖x - Xa‖_2`.

    """
    def _covar(x_):
        """Actual calculation to be vectorized."""
        Xmat = toeplitz(x_[order-1:-1], x_[order-1::-1])
        xvec = x_[order:]
        # Xmat@coeff = -xvec
        coeff, *__ = np.linalg.lstsq(Xmat, -xvec, rcond=rcond)
        error = np.linalg.norm(xvec + Xmat@coeff)
        return coeff, error

    coeff, error = np.vectorize(_covar, signature="(n)->(l),()")(x)
    return coeff, error


def pcoeff_burg(x, order: int):
    """Burg's method for linear prediction (LP) coefficients.

    .. warning::

       We found this method to be instable, consider using `pcoeff_covar` instead.

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
        Prediction coefficients.
    rho : (...) float np.ndarray
        Variance estimate.

    See Also
    --------
    pcoeff_covar

    References
    ----------
    .. [Kay1988] Kay, S.M., 1988. Modern spectral estimation:
       theory and application. Pearson Education India.

    """
    N = x.shape[-1]
    rho = np.full([*x.shape[:-1], order+1], np.nan)
    rho[..., 0] = _gu_sum(abs(x)**2) / N
    eforw = np.ascontiguousarray(x[..., 1:])
    eback = np.ascontiguousarray(x[..., :-1])
    a = np.full([*x.shape[:-1], order, order], np.nan, dtype=x.dtype)
    for k in range(1, order+1):
        # eq (7.38)
        numer = -2*_gu_sum(eforw*np.conj(eback))
        # eq (7.42) for more efficient denom
        # if k > 1:
        #     denom = (1.0 -  np.abs(aa)**2)*denom - np.abs(eforw)
        # else:
        denom = _gu_sum(abs(eforw)**2) + _gu_sum(abs(eback)**2)
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


def predict(x, pcoeff, num: int, stable=False):
    """Forward-predict a series additional `num` steps.

    Parameters
    ----------
    x : (..., N) complex np.ndarray
        Data of the (time) series to be predicted.
    pcoeff : (..., order) complex np.ndarray
        Prediction coefficients
    num : int
        Number of additional (time) steps.
    stable : bool, optional
        If `stable` exponentially growing terms are suppressed, by setting
        roots outside the unit-circle to zero. (default: False)

    Returns
    -------
    px : (..., N+num) complex np.ndarray
        Data of the (time) series extended by `num` steps, with
        `px[:x.size] = x`.

    See Also
    --------
    pcoeff_covar

    """
    if stable:
        return _predict_stable(x, pcoeff=pcoeff, num=num)

    # Naive implementation of the sum
    start = x.shape[-1]
    order = pcoeff.shape[-1]
    xtended = np.concatenate([x, np.full([*x.shape[:-1], num], np.nan, dtype=x.dtype)], axis=-1)
    for ii in range(start, start+num):
        xtended[..., ii] = -_gu_sum(pcoeff[..., ::-1] * xtended[..., ii-order:ii])
    return xtended


def _predict_stable(x, pcoeff, num: int):
    """Companion matrix implementation removing growing terms.

    The roots of the `pcoeff` polynomial are related to poles in frequency.
    If the root is outside the unit circle, it corresponds to a pole in the
    upper complex half-plane resulting in exponential growth.
    We remove such roots.

    """
    # calculate poles and residues
    order = pcoeff.shape[-1]
    comp_mat = companion(np.r_[1, pcoeff])
    dec = decompose_mat(comp_mat)
    right = dec.rv_inv @ x[..., -order:][..., ::-1]
    left = dec.rv[..., 0, :]
    residues = left*right

    # drop exponential growing terms
    bad = abs(dec.eig) > 1
    # print(f'Bad {np.count_nonzero(bad)}/{bad.size}')
    eig = dec.eig[~bad]
    residues = residues[~bad]

    # predict
    start = x.shape[-1]
    xtended = np.concatenate([x, np.full([*x.shape[:-1], num], np.nan, dtype=x.dtype)], axis=-1)
    for ii in range(start, start+num):
        residues *= eig
        xtended[..., ii] = _gu_sum(residues)
    return xtended


def plot_roots(pcoeff, axis=None):
    """Plot the roots corresponding to `pcoeff`.

    Roots for the forward prediction should be inside the unit-circle.

    Parameters
    ----------
    pcoeff : (order, ) complex np.ndarray
        Prediction coefficients
    axis : matplotlib.axes.Axes , optional
        Axis in which the roots are plotted. (default: ``plt.gca()``)

    Returns
    -------
    axis : matplotlib.axes.Axes
        The `axis` in which the roots were plotted.

    See Also
    --------
    pcoeff_covar

    """
    import matplotlib.pyplot as plt  # pylint: disable=[import-outside-toplevel,import-error]
    if axis is None:
        axis = plt.gca()

    axis.axhline(0, color='dimgray', linewidth=0.8)
    axis.axvline(0, color='dimgray', linewidth=0.8)
    unit_circle = plt.Circle((0, 0), radius=1, edgecolor='black', facecolor='None')
    axis.add_patch(unit_circle)

    comp = companion(np.r_[1, pcoeff])
    eigs = np.linalg.eigvals(comp)
    valid = abs(eigs) <= 1
    axis.scatter(eigs[valid].real, eigs[valid].imag, marker='x')
    axis.scatter(eigs[~valid].real, eigs[~valid].imag, marker='x', color='red')

    axis.set_xlabel(r"$\Re y$")
    axis.set_ylabel(r"$\Im y$")
    return axis
