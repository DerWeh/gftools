# encoding: utf-8
"""Pade analytic continuation for Green's functions and self-energies.

The main aim of this module is to provide analytic continuation based on
averaging over multiple Pade approximates (similar to [1]_).

In most cases the following high level function should be used:

`averaged`, `avg_no_neg_imag`
   Return one-shot analytic continuation evaluated at `z`.

`Averager`
   Returns a function for repeated evaluation of the continued function.

References
----------
.. [1] Schött et al. “Analytic Continuation by Averaging Pade Approximants”.
   Phys Rev B 93, no. 7 (2016): 075104.
   https://doi.org/10.1103/PhysRevB.93.075104.

"""
import logging

from abc import ABC, abstractmethod
from itertools import islice

import numpy as np

from . import Result

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

_PRECISE_TYPES = {np.dtype(np.complex256), np.dtype(np.float128)}


class KindSelector(ABC):
    """Abstract filter class to determine high-frequency behavior of Pade.

    We denote approximants with the corresponding high frequency behavior as
    *valid*.
    Considers all valid approximants including between `n_min` and `n_max`
    Matsubara frequencies.
    """

    @abstractmethod
    def __init__(self, n_min, n_max):
        """Consider approximants including between `n_min` and `n_max` Matsubara frequencies."""
        if n_min < 1:
            raise ValueError(f"`n_min` needs to be at least 1 (n_min: {n_min}).")
        if n_max <= n_min:
            raise ValueError(f"`n_max` ({n_max}) needs to be bigger than `n_min` ({n_min}).")
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
    """Filter approximants such that the high-frequency behavior is :math:`1/ω`.

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
    """Filter approximants such that the high-frequency behavior is a constant.

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
    """Return function to check if imaginary part is smaller than `threshold`.

    This methods is designed to create `valid_pades` for `Averager`.
    The imaginary part of retarded Green's functions and self-energies must be
    negative, this is checked by this filter.
    A threshold is given as Pade overshoots when the function goes sharply to 0.
    See for example the semi-circular spectral function of the Bethe lattice
    with infinite Coordination number as example.
    """
    def filter_neg_imag(z, pade_iter):
        r"""Check which pade approximants have a negative imaginary part.

        Parameters
        ----------
        z : complex ndarray
            The inputpoints at which `pade_iter` is calculated.
        pade_iter : iterable
            Iterator yielding the Pade approximants of shape
            (\*approximant.shape, \*z.shape).

        Returns
        -------
        is_valid : (len(pade_iter), \*approximants.shape) bool ndarray
            True for all approximants that fulfill `apporximant.imag < threshold`.

        """
        axis = tuple(-np.arange(z.ndim) - 1)  # keep axis not corresponding to z
        is_valid = np.array([np.all(pade.imag < threshold, axis=axis) for pade in pade_iter])
        return is_valid
    return filter_neg_imag


def _contains_nan(array) -> bool:
    """Check if `array` contains any NaN."""
    flat = array.reshape(-1)
    return np.isnan(np.dot(flat, flat))


def coefficients(z, fct_z) -> np.ndarray:
    """Calculate the coefficients for the Pade continuation.

    Parameters
    ----------
    z : (N_z, ) complex ndarray
        Array of complex points
    fct_z : (..., N_z) complex ndarray
        Function at points `z`

    Returns
    -------
    coefficients : (..., N_z) complex ndarray
        Array of Pade coefficients, needed to perform Pade continuation.
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
    `_PRECISE_TYPES`. This avoids giving the illusion that the results are more
    precise than the input.

    """
    if z.shape != fct_z.shape[-1:]:
        raise ValueError(f"Dimensions of `z` ({z.shape}) and `fct_z` ({fct_z.shape}) mismatch.")
    res = fct_z.astype(dtype=np.complex256, copy=True)
    for ii in range(z.size - 1):
        res[..., ii+1:] = (res[..., ii:ii+1]/res[..., ii+1:] - 1.)/(z[ii+1:] - z[ii])
    complex_pres = np.complex256 if fct_z.dtype in _PRECISE_TYPES else np.complex
    LOGGER.debug("Input type %s precise: %s -> result type: %s",
                 fct_z.dtype, fct_z.dtype in _PRECISE_TYPES, complex_pres)
    return res.astype(complex_pres, copy=False)


def masked_coefficients(z, fct_z):
    """Calculate coefficients but ignore extreme values.

    Like `coefficients` but probably better for noisy data.
    """
    mat = np.zeros((z.size, *fct_z.shape), dtype=np.complex256)
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
                    print('truncate')
                    mat_pi[jj] = (-1)/(z[jj] - z[last_it])
                else:  # huge quotient
                    print('infty')
                    mat_pi[jj] = np.infty
            last_it = ii
            last_coeff = mat_pi[ii]
        else:
            mask[ii] = False
    LOGGER.info("Number of eliminated coefficients: %s", np.count_nonzero(~mask))
    return mat.diagonal(axis1=0, axis2=-1)


def calc_iterator(z_out, z_in, coeff):
    r"""Calculate Pade continuation of function at points `z_out`.

    The continuation is calculated for different numbers of coefficients taken
    into account, where the number is in [n_min, n_max].
    The algorithm is take from [2]_.

    Parameters
    ----------
    z_out : complex ndarray
        points at with the functions will be evaluated
    z_in : (N_in,) complex ndarray
        complex mesh used to calculate `coeff`
    coeff : (..., N_in) complex ndarray
        coefficients for Pade, calculated from `pade.coefficients`

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
       by Means of N-Point Pade Approximants.” Journal of Low Temperature
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
        multiplier_im = multiplier_im / B2
        B2 = 1 + multiplier_im
        pade, pade_prev = (pade + multiplier_im*pade_prev)/B2, pade

        yield pade.reshape(target_shape)


def Averager(z_in, coeff, *, valid_pades, kind: KindSelector):
    """Create function for averaging Pade scheme.

    Parameters
    ----------
    z_in : (N_in,) complex ndarray
        complex mesh used to calculate `coeff`
    coeff : (..., N_in) complex ndarray
        coefficients for Pade, calculated from `pade.coefficients`
    valid_pades : list_like of bool
        Mask which continuations are correct, all Pades where `valid_pades`
        evaluates to false will be ignored for the average.
    kind : {KindGf, KindSelf}
        Defines the asymptotic of the continued function and the number of
        minumum and maximum input points used for Pade. For `KindGf` the
        function goes like :math:`1/z` for large `z`, for `KindSelf` the
        function behaves like a constant for large `z`.

    Returns
    -------
    average : function
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
        raise TypeError(f"Invalid type of `valid_pades`: {valid_pades.dtype}\n"
                        "Expected `bool`.")
    if not valid_pades.any(axis=0).all():
        # for some axis no valid pade was found
        raise RuntimeError("No Pade fulfills is valid.\n"
                           f"No solution found for coefficient (shape: {coeff.shape[:-1]}) axes "
                           f"{np.argwhere(~valid_pades.any(axis=0))}")
    LOGGER.info("Number of valid Pade approximants: %s", np.count_nonzero(valid_pades, axis=0))

    def average(z) -> Result:
        """Calculate Pade continuation of function at points `z`.

        The continuation is calculated for different numbers of coefficients
        taken into account, where the number is in [n_min, n_max]. The function
        value es well as its variance is calculated. The variance should not be
        confused with an error estimate.

        Parameters
        ----------
        z : complex ndarray
            points at with the functions will be evaluated

        Returns
        -------
        pade.x : complex ndarray
            function evaluated at points `z`
        pade.err : complex ndarray
            variance associated with the function values `pade.x` at points `z`

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
                raise RuntimeError("Calculation of Pades failed, results contains NaNs")
        else:
            pades = np.array(list(pade_iter))
            if _contains_nan(pades):
                raise RuntimeError("Calculation of Pades failed, results contains NaNs")
            pades[~valid_pades] = np.nan + 1j*np.nan

        pade_avg = np.nanmean(pades, axis=0)
        std = np.nanstd(pades.real, axis=0, ddof=1) + 1j*np.nanstd(pades.imag, axis=0, ddof=1)

        return Result(x=pade_avg, err=std)
    return average


def Mod_Averager(z_in, coeff, mod_fct, *, valid_pades, kind: KindSelector, vectorized=True):
    r"""Create function for averaging Pade scheme using `mod_fct` before the average.

    This function behaves like `Averager` just that `mod_fct` is applied before
    taking the averages. This should be used, if not the analytic continuation
    but a mollification thereof is used.

    Parameters
    ----------
    z_in : (N_in,) complex ndarray
        complex mesh used to calculate `coeff`
    coeff : (..., N_in) complex ndarray
        coefficients for Pade, calculated from `pade.coefficients`
    mod_fct : callable
        Modification of the analytic continuation. The signature of the function
        should be `mod_fct` (z, pade_z, \*args, \*\*kwds), the tow first
        arguments are the point of evaluation `z` and the single Pade approximants.
    valid_pades : list_like of bool
        Mask which continuations are correct, all Pades where `valid_pades`
        evaluates to false will be ignored for the average.
    kind : {KindGf, KindSelf}
        Defines the asymptotic of the continued function and the number of
        minumum and maximum input points used for Pade. For `KindGf` the
        function goes like :math:`1/z` for large `z`, for `KindSelf` the
        function behaves like a constant for large `z`.
    vectorized : bool, optional
        If `vectorized`, all approximants are given to the function simultaniously
        where the first dimension corresponds to the approximants.
        If not `vectorized`, `mod_fct` will be called for every approximant
        seperately. (default: True)

    Returns
    -------
    mod_average : function
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
        raise TypeError(f"Invalid type of `valid_pades`: {valid_pades.dtype}\n"
                        "Expected `bool`.")
    if not valid_pades.any(axis=0).all():
        # for some axis no valid pade was found
        raise RuntimeError("No Pade fulfills is valid.\n"
                           f"No solution found for coefficient (shape: {coeff.shape[:-1]}) axes "
                           f"{np.argwhere(~valid_pades.any(axis=0))}")
    LOGGER.info("Number of valid Pade approximants: %s", np.count_nonzero(valid_pades, axis=0))

    def mod_average(z, *args, **kwds) -> Result:
        """Calculate modified Pade continuation of function at points `z`.

        Calculate the averaged continuation of `mod_fct(f_z, *args, **kwds)`
        The continuation is calculated for different numbers of coefficients
        taken into account, where the number is in [n_min, n_max]. The function
        value es well as its variance is calculated. The variance should not be
        confused with an error estimate.

        Parameters
        ----------
        z : complex ndarray
            points at with the functions will be evaluated
        args, kwds :
            Passed to the `mod_fct` {mod_fct.__name__}.

        Returns
        -------
        pade.x : complex ndarray
            function evaluated at points `z`
        pade.err : complex ndarray
            variance associated with the function values `pade.x` at points `z`

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
                raise RuntimeError("Calculation of Pades failed, results contains NaNs")
        else:
            pades = np.array(list(pade_iter))
            if _contains_nan(pades):
                raise RuntimeError("Calculation of Pades failed, results contains NaNs")
            pades[~valid_pades] = np.nan + 1j*np.nan

        if vectorized:
            mod_pade = mod_fct(z, pades, *args, **kwds)
        else:
            mod_pade = np.array([mod_fct(z, pade_ii, *args, **kwds)
                                 for pade_ii in pades])
        pade_avg = np.nanmean(mod_pade, axis=0)
        # define helper pade_std np.nanstd( ,axis=0, ddof=1) if complex...
        if np.iscomplexobj(mod_pade):
            std = np.nanstd(mod_pade.real, axis=0, ddof=1) + 1j*np.nanstd(mod_pade.imag, axis=0, ddof=1)
        else:
            std = np.nanstd(mod_pade, axis=0, ddof=1)

        return Result(x=pade_avg, err=std)
    mod_average.__doc__ = mod_average.__doc__.format(mod_fct=mod_fct)
    return mod_average


def averaged(z_out, z_in, *, valid_z=None, fct_z=None, coeff=None,
             filter_valid=None, kind: KindSelector):
    """Return the averaged Pade continuation with its variance.

    The output is checked to have an imaginary part smaller than `threshold`,
    as retarded Green's functions and self-energies have a negative imaginary
    part.
    This is a helper to conveniently get the continuation, it comes however with
    overhead.

    Parameters
    ----------
    z_out : (N_out,) complex ndarray
        points at with the functions will be evaluated
    z_in : (N_in,) complex ndarray
        complex mesh used to calculate `coeff`
    valid_z : (N_out,) complex ndarray, optional
        The output range according to which the Pade approximation is validated
        (compared to the `threshold`).
    fct_z : (N_z, ) complex ndarray, optional
        Function at points `z` from which the coefficients will be calculated.
        Can be omitted if `coeff` is directly given.
    coeff : (N_in,) complex ndarray, optional
        Coefficients for Pade, calculated from `pade.coefficients`. Can be given
        instead of `fct_z`.
    filter_valid : callable
        Function determining which approximants to keep. The signature should
        be filter_valid(ndarray, iterable) -> bool ndarray.
        Currently there are the functions {`FilterNegImag`, } implemented
        to generate filter functions. Look into the implemented for details
        to create new filters.
    kind : {KindGf, KindSelf}
        Defines the asymptotic of the continued function and the number of
        minumum and maximum input points used for Pade. For `KindGf` the
        function goes like :math:`1/z` for large `z`, for `KindSelf` the
        function behaves like a constant for large `z`.

    Returns
    -------
    averaged.x : (N_in, N_out) complex ndarray
        function evaluated at points `z`
    averaged.err : (N_in, N_out) complex ndarray
        variance associated with the function values `pade.x` at points `z`

    """
    assert fct_z is None or coeff is None
    if fct_z is not None:
        coeff = coefficients(z_in, fct_z=fct_z)
    if valid_z is None:
        valid_z = z_out
    if filter_valid is not None:
        validity_iter = kind.islice(calc_iterator(valid_z, z_in, coeff=coeff))
        is_valid = filter_valid(valid_z, validity_iter)
    else:
        is_valid = np.ones((len(kind), *coeff.shape[:-1]), dtype=bool)

    assert is_valid.shape[1:] == coeff.shape[:-1]

    _average = Averager(z_in, coeff=coeff, valid_pades=is_valid, kind=kind)
    return _average(z_out)


def avg_no_neg_imag(z_out, z_in, *, valid_z=None, fct_z=None, coeff=None,
                    threshold=1e-8, kind: KindSelector):
    """Average Pade filtering approximants with non-negative imaginary part.

    This function wraps `averaged`, see `averaged` for the parameters.

    Other Parameters
    ----------------
    threshold : float, optional
        The numerical threshold, how large of an positive imaginary part is
        tolerated (default: 1e-8). `np.infty` can be given to accept all.

    Returns
    -------
    averaged.x : (N_in, N_out) complex ndarray
        function evaluated at points `z`
    averaged.err : (N_in, N_out) complex ndarray
        variance associated with the function values `pade.x` at points `z`

    """
    filter_neg_imag = FilterNegImag(threshold)
    return averaged(z_out=z_out, z_in=z_in, valid_z=valid_z, fct_z=fct_z,
                    coeff=coeff, filter_valid=filter_neg_imag, kind=kind)

# def SelectiveAverage(object):
#     """Do not accept Matsubara frequencies, which make the result unphysical."""