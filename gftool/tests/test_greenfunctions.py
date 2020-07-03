# coding: utf8
"""Tests for Green's functions and related functions.

TODO: use accuracy of *integrate.quad* for *pytest.approx*
TODO: explicit add imaginary axis to the mesh
TODO: make use of the fact, that gf(w>0)=gf_ret(w), gf(w<0)=gf_adv(w)
"""
from __future__ import absolute_import, unicode_literals

from functools import wraps, partial

import pytest
from hypothesis import assume, given, strategies as st
from hypothesis_gufunc.gufunc import gufunc_args

import numpy as np
import scipy.integrate as integrate

import mpmath
from mpmath import fp

from .context import gftool as gt


nonneg_float = st.floats(min_value=0.)
pos_float = st.floats(min_value=0., exclude_min=True)


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

    gf = method(gt.bethe_gf_z)

    @pytest.fixture(params=[0.7, 1.2, ])
    def params(self, request):
        """Parameters for Bethe Green's function."""
        return (), {'half_bandwidth': request.param}


class TestOnedimGf(GfProperties):
    """Check properties of one-dimensional Gf."""

    D = 1.2
    z_mesh = np.mgrid[-2*D:2*D:5j, -2*D:2*D:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gt.onedim_gf_z)

    @pytest.fixture(params=[0.7, 1.2])
    def params(self, request):
        """Parameters for Bethe Green's function."""
        return (), {'half_bandwidth': request.param}


class TestSquareGf(GfProperties):
    """Check properties of Bethe Gf."""

    D = 1.2
    z_mesh = np.mgrid[-2*D:2*D:5j, -2*D:2*D:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gt.square_gf_z)

    @pytest.fixture(params=[0.7, 1.2, ])
    def params(self, request):
        """Parameters for Bethe Green's function."""
        return (), {'half_bandwidth': request.param}


class TestSurfaceGf(GfProperties):
    """Check properties of surface Gf."""

    z_mesh = np.mgrid[-2:2:5j, -2:2:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gt.surface_gf_zeps)

    @pytest.fixture(params=[-.8, -.4, 0., .5, .7])
    def params(self, request):
        """Parameters for the Surface Bethe Green's function."""
        return (), {'eps': request.param,
                    'hopping_nn': .2,
                    }

    def band_edges(self, params):
        """Bandages are shifted ones of `gt.bethe_gf_z`."""
        hopping_nn = params[1]['hopping_nn']
        eps = params[1]['eps']
        return -2*hopping_nn-abs(eps), 2*hopping_nn+abs(eps)


class TestHubbardDimer(GfProperties):
    """Check properties of Hubbard Dimer Gf."""

    z_mesh = np.mgrid[-2:2:5j, -2:2:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gt.hubbard_dimer_gf_z)

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
@given(z=st.complex_numbers(allow_infinity=False, max_magnitude=1e8))  # quad doesn't handle inf
def test_bethe_derivative_1(z, D):
    """Check derivative against numerical solution."""
    assume(z.imag != 0)  # Gf have poles on real axis
    with mpmath.workdps(30):  # improved integration accuracy in case of large inter
        gf_d1 = fp.diff(partial(gt.bethe_gf_z, half_bandwidth=D), z,
                        method='quad', radius=z.imag/2)
    assert np.allclose(gf_d1, gt.bethe_gf_d1_z(z, half_bandwidth=D))


@pytest.mark.parametrize("D", [0.5, 1., 2.])
@given(z=st.complex_numbers(allow_infinity=False, max_magnitude=1e8))  # quad doesn't handle inf
def test_bethe_derivative_2(z, D):
    """Check derivative against numerical solution."""
    assume(z.imag != 0)  # Gf have poles on real axis
    fct = partial(gt.bethe_gf_d1_z, half_bandwidth=D)
    fct_d1 = partial(gt.bethe_gf_d2_z, half_bandwidth=D)
    with mpmath.workdps(30):  # improved integration accuracy in case of large inter
        gf_d1 = fp.diff(fct, z, method='quad', radius=z.imag/2)
    assert np.allclose(gf_d1, fct_d1(z))


@pytest.mark.parametrize("D", [0.5, 1., 2.])
def test_dos_unit(D):
    """Integral over the whole DOS should be 1."""
    assert integrate.quad(gt.bethe_dos, -D-.1, D+.1, args=(D,))[0] == pytest.approx(1.)


@pytest.mark.parametrize("D", [0.5, 1., 2.])
def test_dos_half(D):
    """DOS should be symmetric -> integral over the half should yield 0.5."""
    assert integrate.quad(gt.bethe_dos, -D-.1, 0., args=(D,))[0] == pytest.approx(.5)
    assert integrate.quad(gt.bethe_dos, 0., D+.1, args=(D,))[0] == pytest.approx(.5)


def test_dos_support():
    """DOS should have no support for | eps | > D."""
    D = 1.2
    for eps in np.linspace(D + 1e-6, D*1e4):
        assert gt.bethe_dos(eps, D) == 0
        assert gt.bethe_dos(-eps, D) == 0


def test_imag_gf_negative():
    """Imaginary part of Gf must be smaller or equal 0 for real frequencies."""
    D = 1.2
    omega, omega_step = np.linspace(-D, D, dtype=np.complex, retstep=True)
    omega += 5j*omega_step
    assert np.all(gt.bethe_gf_z(omega, D).imag <= 0)


def test_imag_gf_equals_dos():
    r"""Imaginary part of the GF is proportional to the DOS.

    .. math::
        DOS(ϵ) = -ℑ(G(ϵ))/π
    """
    D = 1.2
    num = int(1e6)
    omega = np.linspace(-D, D, dtype=np.complex, num=num)
    omega += 1j*1e-16
    assert np.allclose(-gt.bethe_gf_z(omega, D).imag/np.pi,
                       gt.bethe_dos(omega, D))


def test_hilbert_equals_integral():
    """Compare *bethe_hilbert_transform* with explicit calculation of integral.

    The integral is singular for xi=0, actually the Cauchy principal value
    should be taken.
    """
    D = 1.2
    xi_mesh = np.mgrid[-2*D:2*D:4j, -2*D:2*D:4j]
    xi_values = np.ravel(xi_mesh[0] + 1j*xi_mesh[1])

    def kernel(eps, xi):
        """Integrand for the Hilbert transform."""
        return gt.bethe_dos(eps, half_bandwidth=D)/(xi - eps)

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
        assert gt.bethe_hilbert_transform(xi, D) == pytest.approx(compare)


@pytest.mark.parametrize("D", [0.5, 1.7, 2.])
def test_bethe_dos_moment(D):
    """Moment is integral over ϵ^m DOS."""
    # check influence of bandwidth, as they are calculated for D=1 and normalized
    m2 = fp.quad(lambda eps: eps**2 * gt.bethe_dos(eps, half_bandwidth=D), [-D, 0, D])
    m3 = fp.quad(lambda eps: eps**3 * gt.bethe_dos(eps, half_bandwidth=D), [-D, 0, D])
    m4 = fp.quad(lambda eps: eps**4 * gt.bethe_dos(eps, half_bandwidth=D), [-D, 0, D])
    assert gt.bethe_dos.m2(D) == pytest.approx(m2)
    assert gt.bethe_dos.m3(half_bandwidth=D) == pytest.approx(m3)
    assert gt.bethe_dos.m4(half_bandwidth=D) == pytest.approx(m4)


@pytest.mark.parametrize("D", [0.5, 1., 2.])
def test_onedim_dos_unit(D):
    """Integral over the whole DOS should be 1."""
    dos = partial(gt.onedim_dos, half_bandwidth=D)
    assert fp.quad(dos, [-D, D]) == pytest.approx(1.)


@pytest.mark.parametrize("D", [0.5, 1., 2.])
def test_onedim_dos_half(D):
    """DOS should be symmetric -> integral over the half should yield 0.5."""
    dos = partial(gt.onedim_dos, half_bandwidth=D)
    assert fp.quad(dos, [-D, 0.]) == pytest.approx(.5)
    assert fp.quad(dos, [0., +D]) == pytest.approx(.5)


def test_onedim_dos_support():
    """DOS should have no support for | eps | > D."""
    D = 1.2
    for eps in np.linspace(D + 1e-6, D*1e4):
        assert gt.onedim_dos(eps, D) == 0
        assert gt.onedim_dos(-eps, D) == 0


@pytest.mark.parametrize("D", [0.5, 1.7, 2.])
def test_onedim_dos_moment(D):
    """Moment is integral over ϵ^m DOS."""
    # check influence of bandwidth, as they are calculated for D=1 and normalized
    m2 = fp.quad(lambda eps: eps**2 * gt.onedim_dos(eps, half_bandwidth=D), [-D, D])
    m3 = fp.quad(lambda eps: eps**3 * gt.onedim_dos(eps, half_bandwidth=D), [-D, D])
    m4 = fp.quad(lambda eps: eps**4 * gt.onedim_dos(eps, half_bandwidth=D), [-D, D])
    assert gt.onedim_dos.m2(D) == pytest.approx(m2)
    assert gt.onedim_dos.m3(half_bandwidth=D) == pytest.approx(m3)
    assert gt.onedim_dos.m4(half_bandwidth=D) == pytest.approx(m4)


@pytest.mark.parametrize("D", [0.5, 1., 2.])
def test_square_dos_unit(D):
    """Integral over the whole DOS should be 1."""
    dos = partial(gt.square_dos, half_bandwidth=D)
    assert fp.quad(dos, [-D, 0., D]) == pytest.approx(1.)


@pytest.mark.parametrize("D", [0.5, 1., 2.])
def test_square_dos_half(D):
    """DOS should be symmetric -> integral over the half should yield 0.5."""
    dos = partial(gt.square_dos, half_bandwidth=D)
    assert fp.quad(dos, [-D, 0.]) == pytest.approx(.5)
    assert fp.quad(dos, [0., +D]) == pytest.approx(.5)


def test_square_dos_support():
    """DOS should have no support for | eps | > D."""
    D = 1.2
    for eps in np.linspace(D + 1e-6, D*1e4):
        assert gt.square_dos(eps, D) == 0
        assert gt.square_dos(-eps, D) == 0


@pytest.mark.parametrize("D", [0.5, 1.7, 2.])
def test_square_dos_moment(D):
    """Moment is integral over ϵ^m DOS."""
    # check influence of bandwidth, as they are calculated for D=1 and normalized
    m2 = fp.quad(lambda eps: eps**2 * gt.square_dos(eps, half_bandwidth=D), [-D, 0, D])
    m3 = fp.quad(lambda eps: eps**3 * gt.square_dos(eps, half_bandwidth=D), [-D, 0, D])
    m4 = fp.quad(lambda eps: eps**4 * gt.square_dos(eps, half_bandwidth=D), [-D, 0, D])
    assert gt.square_dos.m2(D) == pytest.approx(m2)
    assert gt.square_dos.m3(half_bandwidth=D) == pytest.approx(m3)
    assert gt.square_dos.m4(half_bandwidth=D) == pytest.approx(m4)


@pytest.mark.filterwarnings("ignore:(invalid value)|(overflow)|(divide by zero):RuntimeWarning")
@given(gufunc_args('(),(N),(N)->()',
                   dtype=[np.complex_, np.float_, np.float_],
                   elements=[st.complex_numbers(), st.floats(), st.floats()],
                   max_dims_extra=3)
       )
def test_pole_gf_z_gu(args):
    """Check that `gt.pole_gf_z` is a proper gu-function and ensure symmetry."""
    z, poles, weights = args
    assert np.allclose(np.conjugate(gt.pole_gf_z(z, poles=poles, weights=weights)),
                       gt.pole_gf_z(np.conjugate(z), poles=poles, weights=weights),
                       equal_nan=True)


@pytest.mark.filterwarnings("ignore:(invalid value)|(overflow):RuntimeWarning")
@given(gufunc_args('(),(N),(N),()->()',
                   dtype=[np.float_, np.float_, np.float_, np.float_],
                   elements=[st.floats(min_value=0., max_value=1.), st.floats(), nonneg_float, nonneg_float],
                   max_dims_extra=3)
       )
def test_pole_gf_tau_gu(args):
    """Check that `gt.pole_gf_tau` is a proper gu-function and ensure negativity."""
    tau, poles, weights, beta = args
    tau = tau * beta
    assume(not np.any(np.isnan(tau)))
    gf_tau = gt.pole_gf_tau(tau, poles=poles, weights=weights, beta=beta)
    gf_tau = np.nan_to_num(gf_tau, -1)  # nan is valid result
    assert np.all(-1*weights.sum() <= gf_tau) and np.all(gf_tau <= 0)


@pytest.mark.filterwarnings("ignore:(invalid value)|(overflow)|(devide by zero):RuntimeWarning")
@given(gufunc_args('(),(N),(N),()->()',
                   dtype=[np.float_, np.float_, np.float_, np.float_],
                   elements=[st.floats(min_value=0., max_value=1.), pos_float, nonneg_float, pos_float],
                   max_dims_extra=3)
       )
def test_pole_gf_tau_b_gu(args):
    """Check that `gt.pole_gf_tau_b` is a proper gu-function and ensure negativity."""
    tau, poles, weights, beta = args
    tau = tau * beta
    assume(not np.any(np.isnan(tau)))
    assume(np.all(poles*np.asanyarray(beta)[..., np.newaxis] > 0))
    gf_tau = gt.pole_gf_tau_b(tau, poles=poles, weights=weights, beta=beta)
    gf_tau = np.nan_to_num(gf_tau, -1)  # nan is valid result
    assert np.all(gf_tau <= 0)


def test_square_stress_trafo():
    """Compare `stress_trafo` against numerical integration for a selection of points."""
    def stress_tensor(eps, half_bandwidth):
        return -0.5 * eps * gt.square_dos(eps, half_bandwidth=half_bandwidth)

    zz_points = [
        0.371 + 0.1075j,
        0.371 - 0.1075j,
        3.1 + 1e-6j,
        -3 + 1e-6j
    ]
    D = 1.17

    for zz in zz_points:
        # pylint: disable=cell-var-from-loop
        with mpmath.workdps(30):
            integ = fp.quad(lambda eps: stress_tensor(eps, half_bandwidth=D)/(zz - eps), [-D, 0, D])
        assert np.allclose(gt.lattice.square.stress_trafo(zz, half_bandwidth=D), integ)
