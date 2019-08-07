# coding: utf8
"""Tests for Green's functions and related functions.

TODO: use accuracy of *integrate.quad* for *pytest.approx*
TODO: explicit add imaginary axis to the mesh
TODO: make use of the fact, that gf(w>0)=gf_ret(w), gf(w<0)=gf_adv(w)
"""
from __future__ import absolute_import, unicode_literals

from functools import wraps

import pytest

import numpy as np
import scipy.integrate as integrate

from mpmath import fp

from .context import gftools


def method(func):
    """Perpend `self` to `func` to turn it into a method."""
    @wraps(func)
    def wrapper(__, *args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


class GfProperties(object):
    r"""Generic class to test basic properties of a fermionic Gf :math:`G(z)`.

    Checks the analytical properties a one particle Gf of the structure

    .. math::
        G_{ii}(z) = -⟨c_i(z) c_i^†(z)⟩.

    Here `i` can be any quantum number.
    Look into https://gist.github.com/abele/ee049b1fdf7e4a1af71a
    """

    z_mesh: np.ndarray  # mesh on which the function's properties will be tested
    s = +1  # Fermions

    def gf(self, z, **kwargs):
        """signature: gf(z: array(complex), ** kwargs) -> array(complex)."""
        raise NotImplementedError('This is just a placeholder')

    @pytest.fixture
    def params(self):
        """Contains possible parameters needed for the Green's function."""
        return (), {}

    def band_edges(self, params):
        """Return the support of the Green's function, by default (-∞, ∞).

        Can be overwritten by subclasses using the `params`.
        """
        return -np.infty, np.infty

    @pytest.mark.skip(reason="This test probably is wrong!")
    def test_symmetry(self, params):
        r""":math:`G_{ij}(-z) = -s G_ji(z)`.

        FIXME: this is not correct!!!
        As we have :math:`i=j` we get anti-symmetry for fermions
        `G_{ii}(-z) = -s G_ii(z)`.
        """
        assert np.allclose(self.gf(-self.z_mesh, *params[0], **params[1]),
                           -self.s * self.gf(self.z_mesh, *params[0], **params[1]))

    def test_complex(self, params):
        r""":math:`G_{AB}^*(z) = G_{B^† A^†}(z^*)`."""
        assert np.allclose(np.conjugate(self.gf(self.z_mesh, *params[0], **params[1])),
                           self.gf(np.conjugate(self.z_mesh), *params[0], **params[1]))

    def test_limit(self, params):
        r""":math:`\lim_{z→∞} zG(z) = 1`."""
        assert np.allclose(  # along real axis
            fp.limit(lambda zz: zz*self.gf(zz, *params[0], **params[1]).real, np.infty), 1,
            rtol=1e-2
        )
        assert np.allclose(  # along imaginary axis
            fp.limit(lambda zz: -zz*self.gf(1j*zz, *params[0], **params[1]).imag, np.infty), 1,
            rtol=1e-2
        )

    def test_normalization(self, params):
        r""":math:`-∫dωℑG(ω+iϵ)/π = ∫dϵ ρ(ϵ) = 1`."""
        def dos(omega):
            r"""Wrap the DOS :math:`ρ(ω) = -ℑG(ω+iϵ)/π`."""
            return -self.gf(omega+1e-16j, *params[0], **params[1]).imag/np.pi

        lower, upper = self.band_edges(params)
        assert pytest.approx(integrate.quad(dos, a=lower, b=upper)[0]) == 1.


class TestBetheGf(GfProperties):
    """Check properties of Bethe Gf."""

    D = 1.2
    z_mesh = np.mgrid[-2*D:2*D:5j, -2*D:2*D:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gftools.bethe_gf_omega)

    @pytest.fixture(params=[0.7, 1.2, ])
    def params(self, request):
        """Parameters for Bethe Green's function."""
        return (), {'half_bandwidth': request.param}


class TestSquareGf(GfProperties):
    """Check properties of Bethe Gf."""

    D = 1.2
    z_mesh = np.mgrid[-2*D:2*D:5j, -2*D:2*D:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gftools.square_gf_omega)

    @pytest.fixture(params=[0.7, 1.2, ])
    def params(self, request):
        """Parameters for Bethe Green's function."""
        return (), {'half_bandwidth': request.param}


class TestBetheSurfaceGf(GfProperties):
    """Check properties of Bethe surface Gf."""

    z_mesh = np.mgrid[-2:2:5j, -2:2:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gftools.bethe_surface_gf)

    @pytest.fixture(params=[-.8, -.4, 0., .5, .7])
    def params(self, request):
        """Parameters for the Surface Bethe Green's function."""
        return (), {'eps': request.param,
                    'hopping_nn': .2,
                    }

    def band_edges(self, params):
        """Bandages are shifted ones of `gftools.bethe_gf_omega`."""
        hopping_nn = params[1]['hopping_nn']
        eps = params[1]['eps']
        return -2*hopping_nn-abs(eps), 2*hopping_nn+abs(eps)


class TestHubbardDimer(GfProperties):
    """Check properties of Hubbard Dimer Gf."""

    z_mesh = np.mgrid[-2:2:5j, -2:2:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gftools.hubbard_dimer_gf_omega)

    @pytest.fixture(params=['+', '-'])
    def params(self, request):
        """Parameters for the Hubbard Dimer Green's function."""
        return (), {'kind': request.param,
                    'hopping': 1.1,
                    'interaction': 1.3,
                    }

    @pytest.mark.skip(reason="Fixing integral: nearly Delta-functions, no band_edges!")
    def test_normalization(self, params):
        raise NotImplementedError

    def test_limit(self, params):
        """Limit of Pols cannot be accurately determined, thus accuracy is reduced."""
        assert np.allclose(  # along real axis
            fp.limit(lambda zz: zz*self.gf(zz, *params[0], **params[1]).real, np.infty), 1,
            rtol=1e-1
        )
        assert np.allclose(  # along imaginary axis
            fp.limit(lambda zz: -zz*self.gf(1j*zz, *params[0], **params[1]).imag, np.infty), 1,
            rtol=1e-2
        )

@pytest.mark.parametrize("D", [0.5, 1., 2.])
def test_dos_unit(D):
    """Integral over the whole DOS should be 1."""
    assert integrate.quad(gftools.bethe_dos, -D-.1, D+.1, args=(D,))[0] == pytest.approx(1.)


@pytest.mark.parametrize("D", [0.5, 1., 2.])
def test_dos_half(D):
    """DOS should be symmetric -> integral over the half should yield 0.5."""
    assert integrate.quad(gftools.bethe_dos, -D-.1, 0., args=(D,))[0] == pytest.approx(.5)
    assert integrate.quad(gftools.bethe_dos, 0., D+.1, args=(D,))[0] == pytest.approx(.5)


def test_dos_support():
    """DOS should have no support for | eps | > D."""
    D = 1.2
    for eps in np.linspace(D + 1e-6, D*1e4):
        assert gftools.bethe_dos(eps, D) == 0
        assert gftools.bethe_dos(-eps, D) == 0


def test_imag_gf_negative():
    """Imaginary part of Gf must be smaller or equal 0 for real frequencies."""
    D = 1.2
    omega, omega_step = np.linspace(-D, D, dtype=np.complex, retstep=True)
    omega += 5j*omega_step
    assert np.all(gftools.bethe_gf_omega(omega, D).imag <= 0)


def test_imag_gf_equals_dos():
    r"""Imaginary part of the GF is proportional to the DOS.

    .. math::
        DOS(ϵ) = -ℑ(G(ϵ))/π
    """
    D = 1.2
    num = int(1e6)
    omega = np.linspace(-D, D, dtype=np.complex, num=num)
    omega += 1j*1e-16
    assert np.allclose(-gftools.bethe_gf_omega(omega, D).imag/np.pi,
                       gftools.bethe_dos(omega, D))


def test_hilbert_equals_integral():
    """Compare *bethe_hilbert_transfrom* with explicit calculation of integral.

    The integral is singular for xi=0, actually the Cauchy principal value
    should be taken.
    """
    D = 1.2
    xi_mesh = np.mgrid[-2*D:2*D:4j, -2*D:2*D:4j]
    xi_values = np.ravel(xi_mesh[0] + 1j*xi_mesh[1])

    def kernel(eps, xi):
        """Integrand for the Hilbert transform."""
        return gftools.bethe_dos(eps, half_bandwidth=D)/(xi - eps)

    def kernel_real(eps, xi):
        """Real part of the integrand."""
        return kernel(eps, xi).real

    def kernel_imag(eps, xi):
        """Real part of the integrand."""
        return kernel(eps, xi).imag

    for xi in xi_values:
        compare = 0
        compare += integrate.quad(kernel_real, -D, D, args=(xi,))[0]
        compare += 1j*integrate.quad(kernel_imag, -D, D, args=(xi,))[0]
        assert gftools.bethe_hilbert_transfrom(xi, D) == pytest.approx(compare)
