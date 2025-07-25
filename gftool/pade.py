"""
Padé analytic continuation for Green's functions and self-energies.

The main aim of this module is to provide analytic continuation based on
averaging over multiple Padé approximants (similar to [1]_).

In most cases the following high level function should be used:

`averaged`, `avg_no_neg_imag`
   Return one-shot analytic continuation evaluated at `z`.

`Averager`
   Returns a function for repeated evaluation of the continued function.

References
----------
.. [1] Schött et al. “Analytic Continuation by Averaging Padé Approximants”.
   Phys Rev B 93, no. 7 (2016): 075104.
   https://doi.org/10.1103/PhysRevB.93.075104.
"""

import logging
from abc import ABC, abstractmethod
from functools import partial
from itertools import islice
from typing import Optional as Opt

import numpy as np

from gftool import Result, _precision
from gftool._util import _gu_sum

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

_complex256 = _precision.complex256
_nan_std = partial(np.nanstd, ddof=1, axis=0)


class KindSelector(ABC):
    """
    Abstract filter class to determine high-frequency behavior of Padé.

    We denote approximants with the corresponding high frequency behavior as
    *valid*.
    Considers all valid approximants including between `n_min` and `n_max`
    Matsubara frequencies.
    """

    @abstractmethod
    def __init__(self, n_min, n_max):
        """Consider approximants including between `n_min` and `n_max` Matsubara frequencies."""
        if n_min < 1:
            msg = f"`n_min` needs to be at least 1 (n_min: {n_min})."
            raise ValueError(msg)
        if n_max <= n_min:
            msg = f"`n_max` ({n_max}) needs to be bigger than `n_min` ({n_min})."
            raise ValueError(msg)
        self.start = n_min - 1  # indices start from 0
        self.stop = n_max - 1  # indices start from 0
        self.step = NotImplemented

    def islice(self, iterable):
        """Return an iterator whose next() method returns valid values from `iterable`."""
        return islice(iterable, self.start, self.stop, self.step)

    @property
    def slice(self):
        """Return slice selecting the valid approximants."""
        return slice(self.start, self.stop, self.step)

    def __getitem__(self, index):
        """Get indices."""
        return range(self.start, self.stop, self.step)[index]

    def __len__(self):
        """Get number of approximants."""
        return len(range(self.start, self.stop, self.step))

    def __repr__(self):
        """Return Meaningful string."""
        return (self.__class__.__name__
                + f'(start={self.start}, stop={self.stop}, step={self.step})')


class KindGf(KindSelector):
    """
    Filter approximants such that the high-frequency behavior is :math:`1/ω`.

    We denote approximants with the corresponding high frequency behavior as
    *valid*.
    Considers all valid approximants including between `n_min` and `n_max`
    Matsubara frequencies.
    """

    def __init__(self, n_min, n_max):
        """Consider approximants including between `n_min` and `n_max` Matsubara frequencies."""
        if not n_min % 2:
            n_min += 1  # odd number for 1/z required
        super().__init__(n_min, n_max)
        self.step = 2


class KindSelf(KindSelector):
    """
    Filter approximants such that the high-frequency behavior is a constant.

    We denote approximants with the corresponding high frequency behavior as
    *valid*.
    Considers all valid approximants including between `n_min` and `n_max`
    Matsubara frequencies.
    """

    def __init__(self, n_min, n_max):
        """Consider approximants including between `n_min` and `n_max` Matsubara frequencies."""
        if n_min % 2:
            n_min += 1  # even number for constant tail required
        super().__init__(n_min, n_max)
        self.step = 2


def FilterNegImag(threshold=1e-8):
    """
    Return function to check if imaginary part is smaller than `threshold`.

    This methods is designed to create `valid_pades` for `Averager`.
    The imaginary part of retarded Green's functions and self-energies must be
    negative, this is checked by this filter.
    A threshold is given as Padé overshoots when the function goes sharply to 0.
    See for example the semi-circular spectral function of the Bethe lattice
    with infinite Coordination number as example.
    """
    def filter_neg_imag(pade_iter):
        r"""
        Check which Padé approximants have a negative imaginary part.

        Parameters
        ----------
        pade_iter : iterable of (..., N_z) complex np.ndarray
            Iterator of analytic continuations as generated by `calc_iterator`.

        Returns
        -------
        (...) bool np.ndarray
            True for all approximants that fulfill `apporximant.imag < threshold`.
        """
        is_valid = np.array([np.all(pade.imag < threshold, axis=-1) for pade in pade_iter])
        LOGGER.debug("Filter Padés with positive imaginary part (threshold: %s): %s",
                     threshold, np.count_nonzero(is_valid, axis=0))
        return is_valid

    return filter_neg_imag


def FilterNegImagNum(abs_num=None, rel_num=None):
    """
    Return function to check how bad the imaginary part gets.

    This methods is designed to create `valid_pades` for `Averager`.
    The imaginary part of retarded Green's functions and self-energies must be
    negative, this is checked by this filter.
    All continuations that are *valid* in this sense are kept, the worst invalid
    are dropped till only `abs_num` remain.

    Warnings
    --------
    Only checked for flat inputs.
    """
    assert abs_num is None or rel_num is None
    assert abs_num is not None or rel_num is not None

    def filter_neg_imag_num(pade_iter):
        badness = np.array([np.max(pade.imag, axis=-1) for pade in pade_iter])
        abs_num_ = rel_num * badness.shape[0] if abs_num is None else abs_num
        if abs_num_ >= badness.shape[0]:
            LOGGER.warning("Skipping filter, not enough Padés (#Padés = %s)",
                           badness.shape[0])
            return np.ones_like(badness, dtype=bool)

        keep = np.argsort(badness, axis=0)[:abs_num_]  # abs_num_ best results
        is_valid = np.zeros_like(badness, dtype=bool)
        is_valid[keep] = True
        is_valid[badness <= 0] = True
        LOGGER.debug("Filter Padés with positive imaginary part (keep best %s): %s",
                     abs_num_, np.count_nonzero(is_valid, axis=0))
        assert np.all(np.count_nonzero(is_valid, axis=0) >= abs_num_)
        return is_valid
    return filter_neg_imag_num


def FilterHighVariance(rel_num: Opt[float] = None, abs_num: Opt[int] = None):
    """
    Return function to filter continuations with highest variance.

    Parameters
    ----------
    rel_num : float, optional
        The relative number of continuations to keep.
    abs_num : int, optional
        The absolute number of continuations to keep.

    Returns
    -------
    callable
        The filter function (pade_iter) -> np.ndarray.
    """
    assert abs_num is None or rel_num is None
    assert abs_num is not None or rel_num is not None
    if rel_num is not None:
        assert 0. < rel_num < 1.
    if abs_num is not None:
        assert abs_num > 0

    def filter_high_variance(pade_iter):
        """
        Remove the continuations with highest variance.

        Parameters
        ----------
        pade_iter : iterable of (..., N_z) complex np.ndarray
            Iterator of analytic continuations as generated by `calc_iterator`.

        Returns
        -------
        (...) bool np.ndarray
            Boolean array indicating which continuations to keep.
        """
        pade = np.array(list(pade_iter))
        pade_sum = np.sum(pade, axis=0)
        N_pades = pade.shape[0]
        # iteratively remove Padés with larges deviation
        # why iterative?
        # Awful Padés might give wrong features in average, so it should be corrected
        abs_num_ = int(rel_num*N_pades) if abs_num is None else abs_num
        bad = []  # isin needs list not set
        for nn in range(N_pades, abs_num_, -1):
            diff = nn*pade - pade_sum  # 50% of time
            distance = _gu_sum(diff.real**2 + diff.imag**2)  # 40% of time
            badness = np.argsort(distance, axis=0)[::-1]  # truncate what is not needed
            newbad = badness[np.isin(badness, bad, invert=True)][0]
            bad.append(newbad)
            pade_sum -= pade[newbad]
        try:
            is_valid = np.ones_like(distance, dtype=bool)
        except UnboundLocalError:
            LOGGER.warning("Not enough Padés to filter (#Padés = %s)", N_pades)
            return np.ones_like(pade[..., 0], dtype=bool)
        is_valid[badness[:-abs_num_]] = False
        # assert set(badness[:-abs_num_]) == set(bad)  # FIXME
        return is_valid
    return filter_high_variance


def _contains_nan(array) -> bool:
    """Check if `array` contains any NaN."""
    flat = array.reshape(-1)
    return np.isnan(np.dot(flat, flat))


def coefficients(z, fct_z) -> np.ndarray:
    """
    Calculate the coefficients for the Padé continuation.

    Parameters
    ----------
    z : (N_z, ) complex ndarray
        Array of complex points.
    fct_z : (..., N_z) complex ndarray
        Function at points `z`.

    Returns
    -------
    (..., N_z) complex ndarray
        Array of Padé coefficients, needed to perform Padé continuation.
        Has the same same shape as `fct_z`.

    Raises
    ------
    ValueError
        If the size of `z` and the last dimension of `fct_z` do not match.

    Notes
    -----
    The calculation is always done in quad precision (complex256), as it is
    very sensitive towards rounding errors. Afterwards the type of the result
    is cast back to double precision (complex128) unless the input data of
    `fct_z` was already quad precision {float128, complex256}, see
    `_precision.PRECISE_TYPES`. This avoids giving the illusion that the results are more
    precise than the input.
    """
    if z.shape != fct_z.shape[-1:]:
        msg = f"Dimensions of `z` ({z.shape}) and `fct_z` ({fct_z.shape}) mismatch."
        raise ValueError(msg)
    res = fct_z.astype(dtype=_complex256, copy=True)
    for ii in range(z.size - 1):
        res[..., ii+1:] = (res[..., ii:ii+1]/res[..., ii+1:] - 1.)/(z[ii+1:] - z[ii])
    complex_pres = _complex256 if fct_z.dtype in _precision.PRECISE_TYPES else complex
    LOGGER.debug("Input type %s precise: %s -> result type: %s",
                 fct_z.dtype, fct_z.dtype in _precision.PRECISE_TYPES, complex_pres)
    return res.astype(complex_pres, copy=False)


@partial(np.vectorize, otypes=[complex], signature='(n),(n)->(n)')
def masked_coefficients(z, fct_z):
    """
    Calculate coefficients but ignore extreme values.

    Like `coefficients` but probably better for noisy data.
    """
    mat = np.zeros((z.size, *fct_z.shape), dtype=_complex256)
    mask = np.empty_like(z, dtype=bool)
    mask[:] = True
    # cutoff = 1e-6
    cutoff = 1e-4
    mat[0] = fct_z

    assert np.abs(fct_z[0]) > cutoff
    # last valid
    last_it = 0
    last_coeff = mat[last_it, last_it]

    def signi_diff(element) -> bool:
        """Return if the difference of `element` and `last_coeff` is larger `cutoff`."""
        return abs(last_coeff - element) > cutoff

    def comparable_mag(element) -> bool:
        """Return weather the magnitude of `element` is comparable to `last_coeff`."""
        return abs(last_coeff)/cutoff > abs(element) > abs(last_coeff)*cutoff

    for ii, mat_pi in enumerate(mat[1:], start=1):
        if signi_diff(mat[last_it, ii]) and comparable_mag(mat[last_it, ii]):
            # regular update
            mat_pi[ii] = (last_coeff/mat[last_it, ii] - 1.)/(z[ii] - z[last_it])
            for jj in range(ii+1, z.size):
                if not mask[jj]:
                    continue
                if comparable_mag(mat[last_it, jj]):
                    mat_pi[jj] = (last_coeff/mat[last_it, jj] - 1.)/(z[jj] - z[last_it])
                elif abs(last_coeff) < abs(mat[last_it, jj])*cutoff:  # tiny quotient
                    mat_pi[jj] = (-1)/(z[jj] - z[last_it])
                else:  # huge quotient
                    mat_pi[jj] = np.inf
            last_it = ii
            last_coeff = mat_pi[ii]
        else:
            mask[ii] = False
    LOGGER.info("Number of eliminated coefficients: %s", np.count_nonzero(~mask))
    return mat.diagonal(axis1=0, axis2=-1)


def calc_iterator(z_out, z_in, coeff):
    r"""
    Calculate Padé continuation of function at points `z_out`.

    The continuation is calculated for different numbers of coefficients taken
    into account, where the number is in [n_min, n_max].
    The algorithm is take from [2]_.

    Parameters
    ----------
    z_out : complex ndarray
        Points at with the functions will be evaluated.
    z_in : (N_in,) complex ndarray
        Complex mesh used to calculate `coeff`.
    coeff : (..., N_in) complex ndarray
        Coefficients for Padé, calculated from `pade.coefficients`.

    Yields
    ------
    pade_calc : (..., N_in, z_out.shape) complex np.ndarray
        Function evaluated at points `z_out`.
        numbers of Matsubara frequencies between `n_min` and `n_max`.
        The shape of the elements is the same as `coeff.shape` with the last
        dimension corresponding to N_in replaced by the shape of `z_out`:
        (..., N_in, \*z_out.shape).

    References
    ----------
    .. [2] Vidberg, H. J., and J. W. Serene. “Solving the Eliashberg Equations
       by Means of N-Point Padé Approximants.” Journal of Low Temperature
       Physics 29, no. 3-4 (November 1, 1977): 179-92.
       https://doi.org/10.1007/BF00655090.
    """
    assert coeff.shape[-1] == z_in.size
    target_shape = coeff.shape[:-1] + z_out.shape

    z_out = z_out.reshape(-1)  # accept arbitrary shaped z_out
    id1 = np.ones_like(z_out, dtype=complex)

    pade_prev = 0.*id1
    pade = coeff[..., 0:1]*id1
    B2 = id1

    multiplier = (z_out - z_in[:-1, np.newaxis])*coeff[..., 1:, np.newaxis]
    # move N_in axis in front to iterate over it
    multiplier = np.moveaxis(multiplier, -2, 0).copy()

    for multiplier_im in multiplier:
        mult_im = multiplier_im / B2
        B2 = 1 + mult_im
        pade, pade_prev = (pade + mult_im*pade_prev)/B2, pade

        yield pade.reshape(target_shape)


def Averager(z_in, coeff, *, valid_pades, kind: KindSelector):
    """
    Create function for averaging Padé scheme.

    Parameters
    ----------
    z_in : (N_in,) complex ndarray
        Complex mesh used to calculate `coeff`.
    coeff : (..., N_in) complex ndarray
        Coefficients for Padé, calculated from `pade.coefficients`.
    valid_pades : list_like of bool
        Mask which continuations are correct, all Padés where `valid_pades`
        evaluates to false will be ignored for the average.
    kind : {KindGf, KindSelf}
        Defines the asymptotic of the continued function and the number of
        minimum and maximum input points used for Padé. For `KindGf` the
        function goes like :math:`1/z` for large `z`, for `KindSelf` the
        function behaves like a constant for large `z`.

    Returns
    -------
    function
        The continued function `f(z)` (`z`, ) -> Result. `f(z).x` contains the
        function values `f(z).err` the associated variance.

    Raises
    ------
    TypeError
        If `valid_pades` not of type `bool`
    RuntimeError
        If all there are none elements of `valid_pades` that evaluate to True.
    """
    valid_pades = np.array(valid_pades)
    if valid_pades.dtype != bool:
        msg = (
            f"Invalid type of `valid_pades`: {valid_pades.dtype}\n"
            "Expected `bool`."
        )
        raise TypeError(msg)
    if not valid_pades.any(axis=0).all():
        # for some axis no valid Padé was found
        msg = (
            "No Padé fulfills is valid.\n"
            f"No solution found for coefficient (shape: {coeff.shape[:-1]}) axes "
            f"{np.argwhere(~valid_pades.any(axis=0))}"
        )
        raise RuntimeError(msg)
    LOGGER.info("Number of valid Padé approximants: %s", np.count_nonzero(valid_pades, axis=0))

    def average(z) -> Result:
        """
        Calculate Padé continuation of function at points `z`.

        The continuation is calculated for different numbers of coefficients
        taken into account, where the number is in [n_min, n_max]. The function
        value es well as its variance is calculated. The variance should not be
        confused with an error estimate.

        Parameters
        ----------
        z : complex ndarray
            Points at with the functions will be evaluated.

        Returns
        -------
        pade.x : complex ndarray
            Function evaluated at points `z`.
        pade.err : complex ndarray
            Variance associated with the function values `pade.x` at points `z`.

        Raises
        ------
        RuntimeError
            If the calculated continuation contain any NaNs. This indicates
            invalid input in the coefficients and thus the original function.
        """
        z = np.asarray(z)

        pade_iter = kind.islice(calc_iterator(z, z_in, coeff=coeff))
        if valid_pades.ndim == 1:
            # validity determined for all dimensions -> drop invalid pades
            pades = np.array([pade for pade, valid in zip(pade_iter, valid_pades) if valid])
            if _contains_nan(pades):
                # check if fct_z already contained nans
                msg = "Calculation of Padés failed, results contains NaNs"
                raise RuntimeError(msg)
        else:
            pades = np.array(list(pade_iter))
            if _contains_nan(pades):
                msg = "Calculation of Padés failed, results contains NaNs"
                raise RuntimeError(msg)
            pades[~valid_pades] = np.nan + 1j*np.nan

        pade_avg = np.nanmean(pades, axis=0)
        std = _nan_std(pades.real) + 1j*_nan_std(pades.imag)

        return Result(x=pade_avg, err=std)
    return average


def Mod_Averager(z_in, coeff, mod_fct, *, valid_pades, kind: KindSelector, vectorized=True):
    r"""
    Create function for averaging Padé scheme using `mod_fct` before the average.

    This function behaves like `Averager` just that `mod_fct` is applied before
    taking the averages. This should be used, if not the analytic continuation
    but a mollification thereof is used.

    Parameters
    ----------
    z_in : (N_in,) complex ndarray
        Complex mesh used to calculate `coeff`.
    coeff : (..., N_in) complex ndarray
        Coefficients for Padé, calculated from `pade.coefficients`.
    mod_fct : callable
        Modification of the analytic continuation. The signature of the function
        should be `mod_fct` (z, pade_z, \*args, \*\*kwds), the tow first
        arguments are the point of evaluation `z` and the single Padé approximants.
    valid_pades : list_like of bool
        Mask which continuations are correct, all Padés where `valid_pades`
        evaluates to false will be ignored for the average.
    kind : {KindGf, KindSelf}
        Defines the asymptotic of the continued function and the number of
        minimum and maximum input points used for Padé. For `KindGf` the
        function goes like :math:`1/z` for large `z`, for `KindSelf` the
        function behaves like a constant for large `z`.
    vectorized : bool, optional
        If `vectorized`, all approximants are given to the function simultaneously
        where the first dimension corresponds to the approximants.
        If not `vectorized`, `mod_fct` will be called for every approximant
        separately (default: True).

    Returns
    -------
    function
        The continued function `f(z)` (`z`, ) -> Result. `f(z).x` contains the
        function values `f(z).err` the associated variance.

    Raises
    ------
    TypeError
        If `valid_pades` not of type `bool`
    RuntimeError
        If all there are none elements of `valid_pades` that evaluate to True.
    """
    valid_pades = np.array(valid_pades)
    if valid_pades.dtype != bool:
        msg = (
            f"Invalid type of `valid_pades`: {valid_pades.dtype}\n"
            "Expected `bool`."
        )
        raise TypeError(msg)
    if not valid_pades.any(axis=0).all():
        # for some axis no valid pade was found
        msg = (
            "No Padé fulfills is valid.\n"
            f"No solution found for coefficient (shape: {coeff.shape[:-1]}) axes "
            f"{np.argwhere(~valid_pades.any(axis=0))}"
        )
        raise RuntimeError(msg)
    LOGGER.info("Number of valid Padé approximants: %s", np.count_nonzero(valid_pades, axis=0))

    def mod_average(z, *args, **kwds) -> Result:
        """
        Calculate modified Padé continuation of function at points `z`.

        Calculate the averaged continuation of `mod_fct(f_z, *args, **kwds)`
        The continuation is calculated for different numbers of coefficients
        taken into account, where the number is in [n_min, n_max]. The function
        value es well as its variance is calculated. The variance should not be
        confused with an error estimate.

        Parameters
        ----------
        z : complex ndarray
            Points at with the functions will be evaluated.
        *args, **kwds
            Passed to the `mod_fct` {mod_fct.__name__}.

        Returns
        -------
        pade.x : complex ndarray
            Function evaluated at points `z`.
        pade.err : complex ndarray
            Variance associated with the function values `pade.x` at points `z`.

        Raises
        ------
        RuntimeError
            If the calculated continuation contain any NaNs. This indicates
            invalid input in the coefficients and thus the original function.
        """
        z = np.asarray(z)

        pade_iter = kind.islice(calc_iterator(z, z_in, coeff=coeff))
        if valid_pades.ndim == 1:
            # validity determined for all dimensions -> drop invalid pades
            pades = np.array([pade for pade, valid in zip(pade_iter, valid_pades) if valid])
            if _contains_nan(pades):
                # check if fct_z already contained nans
                msg = "Calculation of Padés failed, results contains NaNs"
                raise RuntimeError(msg)
        else:
            pades = np.array(list(pade_iter))
            if _contains_nan(pades):
                msg = "Calculation of Padés failed, results contains NaNs"
                raise RuntimeError(msg)
            pades[~valid_pades] = np.nan + 1j*np.nan

        if vectorized:
            mod_pade = mod_fct(z, pades, *args, **kwds)
        else:
            mod_pade = np.array([mod_fct(z, pade_ii, *args, **kwds)
                                 for pade_ii in pades])
        pade_avg = np.nanmean(mod_pade, axis=0)
        # define helper pade_std np.nanstd( ,axis=0, ddof=1) if complex...
        if np.iscomplexobj(mod_pade):
            std = _nan_std(mod_pade.real) + 1j*_nan_std(mod_pade.imag)
        else:
            std = _nan_std(mod_pade)

        return Result(x=pade_avg, err=std)
    mod_average.__doc__ = mod_average.__doc__.format(mod_fct=mod_fct)
    return mod_average


def apply_filter(*filters, validity_iter):
    r"""
    Handle usage of filters for Padé.

    Parameters
    ----------
    *filters : callable
        Functions to determine which continuations to keep.
    validity_iter : iterable of (..., N_z) complex np.ndarray
        The iterable of analytic continuations as generated by `calc_iterator`.

    Returns
    -------
    (...) bool np.ndarray
        Array to index which continuations are good.
    """
    if len(filters) == 1:
        return filters[0](validity_iter)
    validity_iter = np.array(list(validity_iter))
    shape = validity_iter.shape
    validity_iter = np.moveaxis(validity_iter, 0, -2).reshape((-1, shape[0], shape[-1]))
    is_valid = np.ones(validity_iter.shape[0:2], dtype=bool)
    for i_valid, i_validity_iter in zip(is_valid, validity_iter):
        for filt in filters:
            is_valid_filt = filt(i_validity_iter[i_valid])
            i_valid[i_valid] = is_valid_filt
            if np.count_nonzero(i_valid) == 0:
                msg = (
                    f"No Padé is valid due to filter {filt}.\n"
                    f"No solution found for coefficient (shape: {validity_iter.shape[1:-1]}) axes "
                    f"{np.argwhere(~is_valid.any(axis=0))}"
                )
                raise RuntimeError(
                    msg
                )
    return np.moveaxis(is_valid, 0, -1).reshape(shape[:-1])


def averaged(z_out, z_in, *, valid_z=None, fct_z=None, coeff=None,
             filter_valid=None, kind: KindSelector):
    """
    Return the averaged Padé continuation with its variance.

    The output is checked to have an imaginary part smaller than `threshold`,
    as retarded Green's functions and self-energies have a negative imaginary
    part.
    This is a helper to conveniently get the continuation, it comes however with
    overhead.

    Parameters
    ----------
    z_out : (N_out,) complex ndarray
        Points at with the functions will be evaluated.
    z_in : (N_in,) complex ndarray
        Complex mesh used to calculate `coeff`.
    valid_z : (N_out,) complex ndarray, optional
        The output range according to which the Padé approximation is validated
        (compared to the `threshold`).
    fct_z : (N_z, ) complex ndarray, optional
        Function at points `z` from which the coefficients will be calculated.
        Can be omitted if `coeff` is directly given.
    coeff : (N_in,) complex ndarray, optional
        Coefficients for Padé, calculated from `pade.coefficients`. Can be given
        instead of `fct_z`.
    filter_valid : callable or iterable of callable
        Function determining which approximants to keep. The signature should
        be filter_valid(iterable) -> bool ndarray.
        Currently there are the functions {`FilterNegImag`, `FilterNegImagNum`,
        `FilterHighVariance`} implemented to generate filter functions.
        Look into the implemented for details to create new filters.
    kind : {KindGf, KindSelf}
        Defines the asymptotic of the continued function and the number of
        minimum and maximum input points used for Padé. For `KindGf` the
        function goes like :math:`1/z` for large `z`, for `KindSelf` the
        function behaves like a constant for large `z`.

    Returns
    -------
    averaged.x : (N_in, N_out) complex ndarray
        Function evaluated at points `z`.
    averaged.err : (N_in, N_out) complex ndarray
        Variance associated with the function values `pade.x` at points `z`.
    """
    assert fct_z is None or coeff is None
    z_in = z_in[:kind.stop]
    coeff = (coefficients(z_in, fct_z=fct_z[..., :kind.stop]) if coeff is None
             else coeff[..., :kind.stop])
    if valid_z is None:
        valid_z = z_out
    valid_z = valid_z.reshape(-1)
    if filter_valid is not None:
        validity_iter = kind.islice(calc_iterator(valid_z, z_in, coeff=coeff))
        try:
            filters = tuple(filter_valid)
        except TypeError:  # only one filter given
            filters = (filter_valid,)
        is_valid = apply_filter(*filters, validity_iter=validity_iter)
    else:
        is_valid = np.ones((len(kind), *coeff.shape[:-1]), dtype=bool)

    assert is_valid.shape[1:] == coeff.shape[:-1]

    _average = Averager(z_in, coeff=coeff, valid_pades=is_valid, kind=kind)
    return _average(z_out)


def avg_no_neg_imag(z_out, z_in, *, valid_z=None, fct_z=None, coeff=None,
                    threshold=1e-8, kind: KindSelector):
    """
    Average Padé filtering approximants with non-negative imaginary part.

    This function wraps `averaged`, see `averaged` for the parameters.

    Returns
    -------
    averaged.x : (N_in, N_out) complex ndarray
        Function evaluated at points `z`.
    averaged.err : (N_in, N_out) complex ndarray
        Variance associated with the function values `pade.x` at points `z`.

    Other Parameters
    ----------------
    threshold : float, optional
        The numerical threshold, how large of an positive imaginary part is
        tolerated (default: 1e-8). `np.infty` can be given to accept all.
    """
    filter_neg_imag = FilterNegImag(threshold)
    return averaged(z_out=z_out, z_in=z_in, valid_z=valid_z, fct_z=fct_z,
                    coeff=coeff, filter_valid=filter_neg_imag, kind=kind)

# def SelectiveAverage(object):
#     """Do not accept Matsubara frequencies, which make the result unphysical."""
