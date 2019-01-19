"""Pade analytic continuation."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from itertools import islice

import numpy as np

from . import Result


# pythran export coefficients(complex[], complex[])
def coefficients(iw, gf_iw):
    """Calculate the coefficients needed to perform the Pade continuation.

    Parameters
    ----------
    iw : complex ndarray
        Array of Matsubara frequencies.
    gf_iw : complex ndarray
        Green's function at the Matsubara frequencies

    Returns
    -------
    coefficients : complex ndarray
        Array of Pade coefficients, needed to perform Pade continuation.

    Raises
    ------
    ValueError
        If the shape of `iw` and `gf_iw` do not match.

    """
    if iw.shape != gf_iw.shape:
        raise ValueError("Dimensions of `iw` and `gf_iw` do not match.")
    mat = np.zeros((iw.size, iw.size), dtype=complex)
    mat[0] = gf_iw
    for ii, mat_pi in enumerate(mat[1:]):
        mat_pi[ii+1:] = (mat[ii, ii]/mat[ii, ii+1:] - 1.)/(iw[ii+1:] - iw[ii])
    return mat.diagonal()


# pythran export calc(complex[], complex[], complex[], int)
def calc(z, iw, coeff, n_max):
    """Calculate Pade continuation of function at points `z`.

    Parameters
    ----------
    z : (N,) complex ndarray
        points at with the functions will be evaluated
    iw : (M,) complex ndarray
        imaginary mesh used to calculate `coeff`
    coeff : (M,) complex ndarray
        coefficients for Pade, calculated from `coefficients`
    n_max : int
        number of imaginary frequencies used for the Pade

    Returns
    -------
    pade_calc : (N,) complex ndarray
        function evaluated at points `z`

    """
    id1 = np.ones_like(z, dtype=complex)
    A0, A1 = 0.*id1, coeff[0]*id1
    B0, B1 = 1.*id1, 1.*id1
    A2, B2 = np.empty_like(z, dtype=complex), np.empty_like(z, dtype=complex)
    for ii in range(1, n_max):
        A2 = A1 + (z - iw[ii-1])*coeff[ii]*A0
        B2 = B1 + (z - iw[ii-1])*coeff[ii]*B0
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
def calc_iterator(z, iw, coeff, n_min, n_max, kind='Gf'):
    """Calculate Pade continuation of function at points `z` for all n<=`n_max`.

    Parameters
    ----------
    z : (N,) complex ndarray
        points at with the functions will be evaluated
    iw : (M,) complex ndarray
        imaginary mesh used to calculate `coeff`
    coeff : (M,) complex ndarray
        coefficients for Pade, calculated from `coefficients`
    n_max : int
        Number of maximum imaginary frequencies used for the Pade

    Returns
    -------
    pade_calc : (N,) complex ndarray
        function evaluated at points `z`

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

    id1 = np.ones_like(z, dtype=complex)

    class State(object):
        __slots__ = ('A0', 'A1', 'A2', 'B2')

        def __init__(self, A0, A1, A2, B2):
            self.A0, self.A1, self.A2, self.B2 = A0, A1, A2, B2

    cs = State(A0=0.*id1, A1=coeff[0]*id1, A2=coeff[0]*id1, B2=id1)

    multiplier = np.subtract.outer(z, iw[:-1])*coeff[1:]
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


def Averager(iw, coeff, n_min, n_max, valid_pades, kind='Gf'):
    assert kind in set(('Gf', 'self'))

    def averaged(z):
        z = np.asarray(z)
        scalar_input = False
        if z.ndim == 0:
            z = z[np.newaxis]
            scalar_input = True

        pade_iter = calc_iterator(z, iw, coeff=coeff, n_min=n_min, n_max=n_max, kind=kind)
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
def averaged(z, iw, n_min, n_max, valid_z, gf_iw=None, coeff=None, threshold=1e-8, kind='Gf'):
    assert gf_iw is None or coeff is None
    if gf_iw is not None:
        coeff = coefficients(iw, gf_iw=gf_iw)

    validity_iter = calc_iterator(valid_z, iw, coeff=coeff, n_min=n_min, n_max=n_max, kind=kind)
    is_valid = (np.all(pade.imag < threshold) for pade in validity_iter)

    _averaged = Averager(iw, coeff=coeff, n_min=n_min, n_max=n_max,
                         valid_pades=is_valid, kind=kind)
    return _averaged(z)

# def Averaged(object):
#     """Pade based on averaging over different number of Matsubara frequencies."""
#     def __init__(self, n_pade_max, validity_range=None, imag_shift=1e-16j, threshold=1e-16):
#         """Define general parameters for Pade.

#         Parameters
#         ----------
#         n_pade_max : int  # FIXME
#             The maximum number of Matsubara frequencies taken into account.
#         validity_range : array(complex), optinal
#             The points for which the analytic continuations must be valid,
#             as defined by the function `is_valid`
#         imag_shift : complex
#             Shift into the upper imaginary half-plane, to broaden the results.
#         threshold : float
#             The criteria how *unphysical* the analytic continuation is allowed to get.
#             See `is_valid`.

#         """
#         if validity_range is None:
#             validity_range = np.linspace(-4., 4., num=1000)
#         self.validity_range = validity_range.astype(np.complex256)
#         self.n_pade_max = n_pade_max
#         self.parameters = np.array(
#             (imag_shift, threshold, np.min(validity_range),
#              np.max(validity_range), len(validity_range)),
#             dtype=[('imag_shift', np.complex),
#                    ('threshold', np.float),
#                    ('omega_check_min', np.complex),
#                    ('omega_check_max', np.complex),
#                    ('omega_check_points', np.int),]
#         )
#         # self.imag_shift = imag_shift
#         # self.threshold = threshold
#         self.pade_results = None

#     def __call__(self, w=None):
#         """Evaluate the analytic continuations at the frequencies w.

#         The result is an average about a range of numbers of Pade points.
#         The object as first to be setup by calling `setup_state` or `load_state`,
#         or manually setting the parameters.

#         Parameters
#         ----------
#         w : array(complex), optional
#             Values at which analytic continuation will be evaluated.
#             Default value is the `validity_range` shifted by `imag_shift`.

#         Returns
#         -------
#         avg : array(complex)
#             The average of the analytical continuations, has the same shape as `w`.
#         ComplexDeviation : tuple(float, float)
#             The standard error of mean for the real and imaginary part of the
#             analytical continuation.

#         """
#         assert self.coefficients is not None
#         if w is None:
#             w = self.validity_range + self.parameters['imag_shift']
#         pades = np.array([self.single_pade(w, n_pade)
#                           for n_pade in self.valid_range])
#         self.pade_results = pades
#         avg = np.average(pades, axis=0)
#         std_real, std_imag = np.std(pades.real, axis=0, ddof=1), np.std(pades.imag, axis=0, ddof=1)
#         return avg, std_real + 1j*std_imag

#     def setup_state(self, iw, gf_iw, n_pade_range):
#         """Prepare the object to calculate Pade for given data.

#         Parameters
#         ----------
#         iw : ndarray(complex)
#             The Matsubara frequencies at which the data `gf_iw` (iw) is available.
#         gf_iw : ndarray(complex)
#             The function on the imaginary axis, which will be continued.
#         n_pade_range : array(int)
#             The range of numbers of Pade points taken into account.

#         """
#         self.calculate_coefficients(iw, gf_iw)
#         self.determin_valid_range(n_pade_range)

#     def calculate_coefficients(self, iw, u):
#         self.coefficients = coefficients(iw, u)
#         self.iw = iw

#     def determin_valid_range(self, n_pade_range):
#         assert self.coefficients is not None
#         valid_range = [_n_pade for _n_pade in n_pade_range
#                        if self.is_valid(_n_pade)]
#         self.valid_range = valid_range
#         return valid_range

#     def is_valid(self, n_pade):
#         omega = self.validity_range + self.parameters["imag_shift"]
#         pade = self.single_pade(omega, n_pade)
#         if np.any(pade.imag > self.parameters['threshold']):
#             return False
#         return True

#     def single_pade(self, omega, n_pade):
#         """Evaluate the Pade for one fixed number of Pade points.

#         Parameters
#         ----------
#         omega : array(complex)
#             Frequency on which the analytical continuation will be evaluated.
#         n_pade : int
#             Number of points taken into account to calculate the analytical continuation.

#         Returns
#         -------
#         single_pade : array(complex)
#             The analytical continuation evaluated at frequencies *omega*.

#         """
#         assert self.coefficients is not None
#         return pade_calc(self.iw, self.coefficients, omega, n_pade)

#     def save_state(self, file):
#         np.savez(file, parameters=self.parameters,
#                  coefficients=self.coefficients,
#                  iw=self.iw,
#                  valid_range=self.valid_range)

#     def load_state(self, file):
#         data = np.load(file)
#         assert np.all(data['parameters'] == self.parameters)
#         valid_range = data['valid_range']
#         mask = valid_range <= self.n_pade_max
#         self.valid_range = valid_range[mask]
#         self.coefficients = data['coefficients']
#         # assert len(self.coefficients) <= self.n_pade_max  # FIXME: too weak
#         self.iw = data['iw']



# def SelectiveAverage(object):
#     """Do not accept Matsubara frequencies, which make the result unphysical."""


if __name__ == '__main__':
    from pathlib import Path
    from sys import path as syspath
    syspath.insert(1, str(Path('.').absolute().parent))
    import gftools as gt
    dir_ = Path('~/workspace/articles/master_thesis/data/m_hfm_m/multilayer/t02_mu045_h-9_UX/U08/1hf_19m_U08_poisson/output/00-G_omega.dat').expanduser()
    test_data = np.loadtxt(dir_, unpack=True)
    beta = 1. / 0.01
    iws = gt.matsubara_frequencies(test_data[0], beta=beta)
    omega = np.linspace(-4, 4., num=1000) + 1e-6j
    gf_mid_iw = test_data[1] + 1j*test_data[2]
    coeff = coefficients(iws, gf_mid_iw)
    pade = np.array([calc(omega, iws, coeff, n_max=n) for n in range(590, 600, 2)])
    pade2 = np.array([pade_ for pade_ in calc_iterator(omega, iws, coeff, n_min=589, n_max=599)])
    pade, std = averaged(omega, iws, gf_mid_iw, n_min=89, n_max=599)
    import matplotlib.pyplot as plt
    plt.errorbar(omega.real, -pade.imag, yerr=std.imag, label=r'$-\Im')
    plt.errorbar(omega.real, pade.real, yerr=std.real, label=r'$\Re$')
