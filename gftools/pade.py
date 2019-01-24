"""Pade analytic continuation for Green's functions and self-energies.

The main aim of this module is to provide analytic continuation based on
averaging over multiple Pade approximates (similar to _[1]).

In most cases the following high level function should be used:

`averaged`
   Returns one-shot analytic continuation evaluated at `z`.

`Averager`
   Returns a function for repeated evaluation of the continued function.

References
----------
.. [1] Schött et al. “Analytic Continuation by Averaging Pade Approximants”.
   Phys Rev B 93, no. 7 (2016): 075104.
   https://doi.org/10.1103/PhysRevB.93.075104.

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from itertools import islice

import numpy as np

from . import Result


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

    """
    if z.shape != fct_z.shape[-1:]:
        raise ValueError(f"Dimensions of `z` ({z.shape}) and `fct_z` ({fct_z.shape}) mismatch.")
    mat = np.zeros((z.size, *fct_z.shape), dtype=complex)
    mat[0] = fct_z
    for ii, mat_pi in enumerate(mat[1:]):
        mat_pi[..., ii+1:] = (mat[ii, ..., ii:ii+1]/mat[ii, ..., ii+1:] - 1.)/(z[ii+1:] - z[ii])
    return mat.diagonal(axis1=0, axis2=-1)


def calc(z_out, z_in, coeff, n_max):
    """Calculate Pade continuation of function at points `z_out`.

    Naive implementation of Vidberg & Serene paper. Not recommended to use!

    Parameters
    ----------
    z_out : (N,) complex ndarray
        points at with the functions will be evaluated
    z_in : (M,) complex ndarray
        complex mesh used to calculate `coeff`
    coeff : (M,) complex ndarray
        coefficients for Pade, calculated from `pade.coefficients`
    n_max : int
        number of `z_in` points used for the Pade

    Returns
    -------
    pade_calc : (N,) complex ndarray
        function evaluated at points `z_out`

    """
    id1 = np.ones_like(z_out, dtype=complex)
    A0, A1 = 0.*id1, coeff[0]*id1
    B0, B1 = 1.*id1, 1.*id1
    A2, B2 = np.empty_like(z_out, dtype=complex), np.empty_like(z_out, dtype=complex)
    for ii in range(1, n_max):
        A2 = A1 + (z_out - z_in[ii-1])*coeff[ii]*A0
        B2 = B1 + (z_out - z_in[ii-1])*coeff[ii]*B0
        A1 /= B2
        A2 /= B2
        B1 /= B2
        B2 /= B2

        A0 = A1
        A1 = A2
        B0 = B1
        B1 = B2
    return A2 / B2


def calc_iterator(z_out, z_in, coeff, n_min, n_max, kind='Gf'):
    """Calculate Pade continuation of function at points `z_out`.

    The continuation is calculated for different numbers of coefficients taken
    into account, where the number is in [n_min, n_max].

    Parameters
    ----------
    z_out : complex ndarray
        points at with the functions will be evaluated
    z_in : (N_in,) complex ndarray
        complex mesh used to calculate `coeff`
    coeff : (..., N_in) complex ndarray
        coefficients for Pade, calculated from `pade.coefficients`
    n_min, n_max : int
        Number of minimum (maximum) input points and coefficients used for Pade
    kind : {'Gf', 'self'}
        Defines the asymptotic of the continued function. For 'Gf' the function
        goes like :math:`1/z` for large `z`, for 'self' the function behaves
        like a constant for large `z`.

    Returns
    -------
    pade_calc : iterator
        Function evaluated at points `z_out` for all corresponding (see `kind`)
        numbers of Matsubara frequencies between `n_min` and `n_max`.
        The shape of the elements is the same as `coeff.shape` with the last
        dimension corresponding to N_in replaced by the shape of `z_out`:
        (..., N_in, *z_out.shape).

    References
    ----------
    .. [2] Vidberg, H. J., and J. W. Serene. “Solving the Eliashberg Equations
       by Means of N-Point Pade Approximants.” Journal of Low Temperature
       Physics 29, no. 3-4 (November 1, 1977): 179-92.
       https://doi.org/10.1007/BF00655090.

    """
    assert kind in set(('Gf', 'self'))
    assert n_min >= 1
    assert n_min < n_max
    n_min -= 1
    n_max -= 1
    if kind == 'Gf' and n_min % 2:
        # odd number for 1/ω required -> index must be even
        n_min += 1
    if kind == 'self' and not n_min % 2:
        # even number for constant tail required -> index must be odd
        n_min += 1
    out_shape = z_out.shape
    coeff_shape = coeff.shape

    z_out = z_out.reshape(-1)  # accept arbitrary shaped z_out
    id1 = np.ones_like(z_out, dtype=complex)

    class State:
        """State of the calculation to emulate nonlocal behavior."""

        __slots__ = ('A0', 'A1', 'A2', 'B2')

        def __init__(self, A0, A1, A2, B2):
            self.A0, self.A1, self.A2, self.B2 = A0, A1, A2, B2

    cs = State(A0=0.*id1, A1=coeff[..., 0:1]*id1, A2=coeff[..., 0:1]*id1, B2=id1)

    multiplier = (z_out - z_in[:-1, np.newaxis])*coeff[..., 1:, np.newaxis]
    # move N_in axis in front to iterate over it
    multiplier = np.moveaxis(multiplier, -2, 0).copy()

    # pythran export calc_iterator._iteration(int)
    def _iteration(multiplier_im):
        multiplier_im = multiplier_im/cs.B2
        cs.A2 = cs.A1 + multiplier_im*cs.A0
        cs.B2 = 1 + multiplier_im

        cs.A0 = cs.A1
        pade = cs.A1 = cs.A2 / cs.B2
        return pade

    complete_iterations = (_iteration(multiplier_im).reshape(*coeff_shape[:-1], *out_shape)
                           for multiplier_im in multiplier)
    return islice(complete_iterations, n_min, n_max, 2)


def Averager(z_in, coeff, n_min, n_max, valid_pades, kind='Gf'):
    """Create function for averaging Pade scheme.

    Parameters
    ----------
    z_in : (N_in,) complex ndarray
        complex mesh used to calculate `coeff`
    coeff : (..., N_in) complex ndarray
        coefficients for Pade, calculated from `pade.coefficients`
    n_min, n_max : int
        Number of minimum (maximum) input points and coefficients used for Pade
    valid_pades : list_like of bool
        Mask which continuations are correct, all Pades where `valid_pades`
        evaluates to false will be ignored for the average.
    kind : {'Gf', 'self'}
        Defines the asymptotic of the continued function. For 'Gf' the function
        goes like :math:`1/z` for large `z`, for 'self' the function behaves
        like a constant for large `z`.

    Returns
    -------
    averaged : function
        The continued function `f(z)` (`z`, ) -> Result. `f(z).x` contains the
        function values `f(z).err` the associated variance.

    Raises
    ------
    TypeError
        If `valid_pades` not of type `bool`
    RuntimeError
        If all there are none elements of `valid_pades` that evaluate to True.

    """
    assert kind in set(('Gf', 'self'))
    valid_pades = np.array(valid_pades)
    if valid_pades.dtype != bool:
        raise TypeError(f"Invalid type of `valid_pades`: {valid_pades.type}\n"
                        "Expected `bool`.")
    if not valid_pades.any(axis=0).all():
        # for some axis no valid pade was found
        raise RuntimeError("No Pade fulfills is valid.\n"
                           f"No solution found for coefficient (shape: {coeff.shape[:-1]}) axes "
                           f"{np.argwhere(~valid_pades.any(axis=0))}")

    def averaged(z) -> Result:
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
        scalar_input = False
        if z.ndim == 0:
            z = z[np.newaxis]
            scalar_input = True

        pade_iter = calc_iterator(z, z_in, coeff=coeff, n_min=n_min, n_max=n_max, kind=kind)
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
            pades[~valid_pades] = np.nan

        pade_avg = np.nanmean(pades, axis=0)
        std = np.nanstd(pades.real, axis=0, ddof=1) + 1j*np.nanstd(pades.real, axis=0, ddof=1)

        if scalar_input:
            return Result(x=np.squeeze(pade_avg, axis=-1), err=np.squeeze(std, axis=-1))
        return Result(x=pade_avg, err=std)
    return averaged


# TODO: make it more abstract, allow to pass a filter function taken a Pade list/iterator
def averaged(z_out, z_in, n_min, n_max, valid_z=None, fct_z=None, coeff=None, threshold=1e-8, kind='Gf'):
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
    n_min, n_max : int
        Number of minimum (maximum) input points and coefficients used for Pade
    valid_z : (N_out,) complex ndarray, optional
        The output range according to which the Pade approximation is validated
        (compared to the `threshold`).
    fct_z : (N_z, ) complex ndarray, optional
        Function at points `z` from which the coefficients will be calculated.
        Can be omitted if `coeff` is directly given.
    coeff : (N_in,) complex ndarray, optional
        Coefficients for Pade, calculated from `pade.coefficients`. Can be given
        instead of `fct_z`.
    threshold : float, optional
        The numerical threshold, how large of an positive imaginary part is
        tolerated (default: 1e-8). `np.infty` can be given to accept all.
    kind : {'Gf', 'self'}
        Defines the asymptotic of the continued function. For 'Gf' the function
        goes like :math:`1/z` for large `z`, for 'self' the function behaves
        like a constant for large `z`.

    Returns
    -------
    averaged.x : (N,) complex ndarray
        function evaluated at points `z`
    averaged.err : (N,) complex ndarray
        variance associated with the function values `pade.x` at points `z`

    """
    assert fct_z is None or coeff is None
    if fct_z is not None:
        coeff = coefficients(z_in, fct_z=fct_z)
    if valid_z is None:
        valid_z = z_out

    validity_iter = calc_iterator(valid_z, z_in, coeff=coeff, n_min=n_min, n_max=n_max, kind=kind)
    is_valid = np.array([np.all(pade.imag < threshold, axis=tuple(-np.arange(valid_z.ndim)-1))
                         for pade in validity_iter])
    assert is_valid.shape[1:] == coeff.shape[:-1]

    _averaged = Averager(z_in, coeff=coeff, n_min=n_min, n_max=n_max,
                         valid_pades=is_valid, kind=kind)
    return _averaged(z_out)

# def SelectiveAverage(object):
#     """Do not accept Matsubara frequencies, which make the result unphysical."""
