"""Pade analytic continuation for Green's functions and self-energies."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from itertools import islice

import numpy as np

from . import Result


# pythran export coefficients(complex[], complex[])
def coefficients(z, fct_z) -> np.ndarray:
    """Calculate the coefficients needed to perform the Pade continuation.

    Parameters
    ----------
    z : (N_z, ) complex ndarray
        Array of complex points
    fct_z : (N_z, ) complex ndarray
        Function at points `z`

    Returns
    -------
    coefficients : (N_z, ) complex ndarray
        Array of Pade coefficients, needed to perform Pade continuation.

    Raises
    ------
    ValueError
        If the shape of `z` and `fct_z` do not match.

    """
    if z.shape != fct_z.shape:
        raise ValueError(f"Dimensions of `z` ({z.shape}) and `fct_z` ({fct_z.shape}) mismatch.")
    mat = np.zeros((z.size, z.size), dtype=complex)
    mat[0] = fct_z
    for ii, mat_pi in enumerate(mat[1:]):
        mat_pi[ii+1:] = (mat[ii, ii]/mat[ii, ii+1:] - 1.)/(z[ii+1:] - z[ii])
    return mat.diagonal()


# pythran export calc(complex[], complex[], complex[], int)
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


# pythran export calc_iterator(complex[], complex[], complex[], int, int)
def calc_iterator(z_out, z_in, coeff, n_min, n_max, kind='Gf'):
    """Calculate Pade continuation of function at points `z_out`.

    The continuation is calculated for different numbers of coefficients taken
    into account, where the number is in [n_min, n_max].

    Parameters
    ----------
    z_out : (N_out,) complex ndarray
        points at with the functions will be evaluated
    z_in : (N_in,) complex ndarray
        complex mesh used to calculate `coeff`
    coeff : (N_in,) complex ndarray
        coefficients for Pade, calculated from `pade.coefficients`
    n_min, n_max : int
        Number of minimum (maximum) input points and coefficients used for Pade
    kind : {'Gf', 'self'}
        Defines the asymptotic of the continued function. For 'Gf' the function
        goes like :math:`1/z` for large `z`, for 'self' the function behaves
        like a constant for large `z`.

    Returns
    -------
    pade_calc : (N_out,) complex ndarray
        function evaluated at points `z_out`

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

    id1 = np.ones_like(z_out, dtype=complex)

    class State(object):
        """State of the calculation to emulate nonlocal behavior."""

        __slots__ = ('A0', 'A1', 'A2', 'B2')

        def __init__(self, A0, A1, A2, B2):
            self.A0, self.A1, self.A2, self.B2 = A0, A1, A2, B2

    cs = State(A0=0.*id1, A1=coeff[0]*id1, A2=coeff[0]*id1, B2=id1)

    multiplier = np.subtract.outer(z_out, z_in[:-1])*coeff[1:]
    multiplier = np.moveaxis(multiplier, -1, 0).copy()

    # pythran export calc_iterator._iteration(int)
    def _iteration(multiplier_im):
        multiplier_im = multiplier_im/cs.B2
        cs.A2 = cs.A1 + multiplier_im*cs.A0
        cs.B2 = 1 + multiplier_im

        cs.A0 = cs.A1
        pade = cs.A1 = cs.A2 / cs.B2
        return pade

    complete_iterations = (_iteration(multiplier_im) for multiplier_im in multiplier)
    return islice(complete_iterations, n_min, n_max, 2)


def Averager(z_in, coeff, n_min, n_max, valid_pades, kind='Gf'):
    """Create function for averaging Pade scheme.

    Parameters
    ----------
    z_in : (N_in,) complex ndarray
        complex mesh used to calculate `coeff`
    coeff : (N_in,) complex ndarray
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

    """
    assert kind in set(('Gf', 'self'))
    valid_pades = list(valid_pades)

    def averaged(z):
        """Calculate Pade continuation of function at points `z`.

        The continuation is calculated for different numbers of coefficients
        taken into account, where the number is in [n_min, n_max]. The function
        value es well as its variance is calculated. The variance should not be
        confused with an error estimate.

        Parameters
        ----------
        z : (N,) complex ndarray
            points at with the functions will be evaluated

        Returns
        -------
        pade.x : (N,) complex ndarray
            function evaluated at points `z`
        pade.err : (N,) complex ndarray
            variance associated with the function values `pade.x` at points `z`

        """
        z = np.asarray(z)
        scalar_input = False
        if z.ndim == 0:
            z = z[np.newaxis]
            scalar_input = True

        pade_iter = calc_iterator(z, z_in, coeff=coeff, n_min=n_min, n_max=n_max, kind=kind)
        pades = np.array([pade for pade, valid in zip(pade_iter, valid_pades) if valid])

        if pades.size == 0:
            raise RuntimeError("No Pade fulfills requirements")
        pade_avg = np.average(pades, axis=0)
        std = np.std(pades.real, axis=0, ddof=1) + 1j*np.std(pades.real, axis=0, ddof=1)

        if scalar_input:
            return Result(x=np.squeeze(pade_avg), err=np.squeeze(std))
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
    is_valid = [np.all(pade.imag < threshold) for pade in validity_iter]

    _averaged = Averager(z_in, coeff=coeff, n_min=n_min, n_max=n_max,
                         valid_pades=is_valid, kind=kind)
    return _averaged(z_out)

# def SelectiveAverage(object):
#     """Do not accept Matsubara frequencies, which make the result unphysical."""
