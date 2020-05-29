"""Different function bases.

The basis classes are based on `NamedTuple`, hence they have hardly any overhead.
On the other hand, no additional checks are performed in the constructor.

"""
from typing import NamedTuple

import numpy as np

from gftool.basis.pole import PoleFct, PoleGf

assert PoleFct, PoleGf


class ZeroPole(NamedTuple):
    """Rational polynomial characterized by zeros and poles.

    The function is given by

    .. math:: ZeroPole.eval(z) = asymptotic * np.prod(z - zeros) / np.prod(z - poles)

    Attributes
    ----------
    zeros, poles : (..., Nz), (..., Np) complex np.ndarray
        Zeros and poles of the represented function.
    amplitude : (...) complex np.ndarray or complex
        The amplitude of the function. This is also the large `abs(z)` limit
        of the function `ZeroPole.eval(z) = amplitude * z**(Nz-Np)`.

    """

    zeros: np.ndarray
    poles: np.ndarray
    amplitude: complex or np.ndarray = 1.

    def eval(self, z):
        """Evaluate the function at `z`.

        Parameters
        ----------
        z : (...) complex np.ndarray

        Returns
        -------
        fct : (...) complex np.ndarray
            The function evaluated at `z`.

        """
        z = np.asanyarray(z)[..., np.newaxis]
        return self.amplitude * np.prod(z - self.zeros, axis=-1) / np.prod(z - self.poles, axis=-1)

    def reciprocal(self, z):
        """Evaluate the reciprocal `1./ZeroPole.eval(z)` at `z`.

        Parameters
        ----------
        z : (...) complex np.ndarray

        Returns
        -------
        recip : (...) complex np.ndarray
            The reciprocal of the function evaluated at `z`.

        """
        z = np.asanyarray(z)[..., np.newaxis]
        numer = np.prod(z - self.poles, axis=-1)
        denom = np.prod(z - self.zeros, axis=-1)
        return 1./self.amplitude * numer / denom

    def to_ratpol(self) -> 'RatPol':
        """Represent the rational polynomial as fraction of two polynomial."""
        numerator = np.polynomial.Polynomial.fromroots(self.zeros)*self.amplitude
        denominator = np.polynomial.Polynomial.fromroots(self.poles)
        return RatPol(numer=numerator, denom=denominator)


class RatPol(NamedTuple):
    """Rational polynomial given as numerator and denominator.

    Attributes
    ----------
    numer, denom : np.polynomial.Polynomial
        Numerator and denominator, given as `numpy` polynomials.

    """

    numer: np.polynomial.Polynomial
    denom: np.polynomial.Polynomial

    def eval(self, z):
        """Evaluate the rational polynomial at `z`."""
        return self.numer(z)/self.denom(z)
