# coding: utf8
"""Tests for Green's functions and related functions.

TODO: use accuracy of *integrate.quad* for *pytest.approx*
TODO: explicit add imaginary axis to the mesh
TODO: make use of the fact, that gf(w>0)=gf_ret(w), gf(w<0)=gf_adv(w)
"""
from functools import partial, wraps
from itertools import product

import mpmath
import pytest
import numpy as np
import scipy.integrate as integrate

from mpmath import fp
from hypothesis import assume, given, strategies as st
from hypothesis_gufunc.gufunc import gufunc_args

from .context import gftool as gt


nonneg_float = st.floats(min_value=0.)
pos_float = st.floats(min_value=0., exclude_min=True)


def method(func):
    """Perpend `self` to `func` to turn it into a method."""
    @wraps(func)
    def wrapper(__, *args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


class GfProperties:
    r"""Generic class to test basic properties of a fermionic Gf :math:`G(z)`.

    Checks the analytical properties a one particle Gf of the structure

    .. math:: G_{ii}(z) = -⟨c_i(z) c_i^†(z)⟩.

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

    def test_normalization(self, params, points=None):
        r""":math:`-∫dωℑG(ω+iϵ)/π = ∫dϵ ρ(ϵ) = 1`."""
        def dos(omega):
            r"""Wrap the DOS :math:`ρ(ω) = -ℑG(ω+iϵ)/π`."""
            return -self.gf(omega+1e-16j, *params[0], **params[1]).imag/np.pi

        lower, upper = self.band_edges(params)
        assert pytest.approx(integrate.quad(dos, a=lower, b=upper, points=points)[0]) == 1.


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
    """Check properties of square Gf."""

    D = 1.2
    z_mesh = np.mgrid[-2*D:2*D:5j, -2*D:2*D:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gt.square_gf_z)

    @pytest.fixture(params=[0.7, 1.2, ])
    def params(self, request):
        """Parameters for Bethe Green's function."""
        return (), {'half_bandwidth': request.param}


class TestRectangularGf(GfProperties):
    """Check properties of rectangular Gf."""

    D = 1.2
    z_mesh = np.mgrid[-2*D:2*D:5j, -2*D:2*D:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gt.lattice.rectangular.gf_z)

    @pytest.fixture(params=[{'half_bandwidth': D, 'scale': gamma}
                            for D, gamma in product([0.7, 1.2], [1.3, 2.0])])
    def params(self, request):
        """Parameters for Bethe Green's function."""
        return (), request.param

    def band_edges(self, params):
        """Return the support of the Green's function."""
        D = params[1]['half_bandwidth']
        return -D, D

    def test_normalization(self, params, points=None):
        """Singularities are needed for accurate integration."""
        del points  # was only give for subclasses
        D, gamma = params[1]['half_bandwidth'], params[1]['scale']
        singularity = D * (gamma - 1) / (gamma + 1)
        super().test_normalization(params, points=[-singularity, singularity])


class TestTriangularGf(GfProperties):
    """Check properties of rectangular Gf."""

    D = 1.2
    z_mesh = np.mgrid[-2*D:2*D:5j, -2*D:2*D:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gt.lattice.triangular.gf_z)

    @pytest.fixture(params=[0.7, 1.2, ])
    def params(self, request):
        """Parameters for Bethe Green's function."""
        return (), {'half_bandwidth': request.param}

    def band_edges(self, params):
        """Return the support of the Green's function."""
        D = params[1]['half_bandwidth']
        return -2*D/3, 4*D/3

    def test_normalization(self, params, points=None):
        """Singularities are needed for accurate integration."""
        del points  # was only give for subclasses
        D = params[1]['half_bandwidth']
        super().test_normalization(params, points=[-4*D/9])


class TestHoneycombGf(GfProperties):
    """Check properties of rectangular Gf."""

    D = 1.2
    z_mesh = np.mgrid[-2*D:2*D:5j, -2*D:2*D:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gt.lattice.honeycomb.gf_z)

    @pytest.fixture(params=[0.7, 1.2, ])
    def params(self, request):
        """Parameters for honeycomb Green's function."""
        return (), {'half_bandwidth': request.param}

    def band_edges(self, params):
        """Return the support of the Green's function."""
        D = params[1]['half_bandwidth']
        return -D, D

    def test_normalization(self, params, points=None):
        """Singularities are needed for accurate integration."""
        del points  # was only give for subclasses
        D = params[1]['half_bandwidth']
        super().test_normalization(params, points=[-D/3, +D/3])


class TestKagomeGf(GfProperties):
    """Check properties of rectangular Gf."""

    D = 1.2
    z_mesh = np.mgrid[-2*D:2*D:5j, -2*D:2*D:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gt.lattice.kagome.gf_z)

    @pytest.fixture(params=[0.7, 1.2, ])
    def params(self, request):
        """Parameters for kagome Green's function."""
        return (), {'half_bandwidth': request.param}

    def band_edges(self, params):
        """Return the support of the Green's function."""
        D = params[1]['half_bandwidth']
        return -2*D/3, 4*D/3

    @pytest.mark.filterwarnings("ignore::UserWarning")  # ignores quad's IntegrationWarning
    def test_normalization(self, params, points=None):
        """Due to dealta-peak we have to go further into the imaginary plane."""
        D = params[1]['half_bandwidth']

        def dos(omega):
            r"""Wrap the DOS :math:`ρ(ω) = -ℑG(ω+iϵ)/π`."""
            omega = omega + 1e-3j
            return -self.gf(omega, *params[0], **params[1]).imag/np.pi

        lower, upper = self.band_edges(params)
        points = np.array([2*lower, lower, 0, 1/3, 2/3, upper, 2*upper]) * D
        assert (integrate.quad(dos, a=10*lower, b=10*upper, points=points)[0]
                == pytest.approx(1, rel=1e-3))


class TestSimpleCubicGf(GfProperties):
    """Check properties of rectangular Gf."""

    D = 1.2
    z_mesh = np.mgrid[-2*D:2*D:5j, -2*D:2*D:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gt.lattice.sc.gf_z)

    @pytest.fixture(params=[0.7, 1.2, ])
    def params(self, request):
        """Parameters for simple cubic Green's function."""
        return (), {'half_bandwidth': request.param}

    def band_edges(self, params):
        """Return the support of the Green's function."""
        D = params[1]['half_bandwidth']
        return -D, D

    def test_normalization(self, params, points=None):
        """Singularities are needed for accurate integration."""
        del points  # was only give for subclasses
        D = params[1]['half_bandwidth']
        super().test_normalization(params, points=[-D/3, D/3])


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
    dos = partial(gt.lattice.bethe.dos, half_bandwidth=D)
    for mm in gt.lattice.bethe.dos_moment_coefficients:
        moment = fp.quad(lambda eps: eps**mm * dos(eps), [-D, +D])
        assert moment == pytest.approx(gt.bethe_dos_moment(mm, half_bandwidth=D))


@given(eps=st.floats(-1.5, +1.5))
def test_bethe_dos_vs_dos_mp(eps):
    """Compare multi-precision and `numpy` implementation of DOS."""
    D = 1.3
    assert np.allclose(gt.bethe_dos(eps, half_bandwidth=D),
                       float(gt.lattice.bethe.dos_mp(eps, half_bandwidth=D)))


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
    dos = partial(gt.lattice.onedim.dos, half_bandwidth=D)
    for mm in gt.lattice.onedim.dos_moment_coefficients:
        moment = fp.quad(lambda eps: eps**mm * dos(eps), [-D, +D])
        assert moment == pytest.approx(gt.onedim_dos_moment(mm, half_bandwidth=D))


@given(eps=st.floats(-1.5, +1.5))
def test_onedim_dos_vs_dos_mp(eps):
    """Compare multi-precision and `numpy` implementation of DOS."""
    D = 1.3
    assert np.allclose(gt.lattice.onedim.dos(eps, half_bandwidth=D),
                       float(gt.lattice.onedim.dos_mp(eps, half_bandwidth=D)))


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
    dos = partial(gt.lattice.square.dos, half_bandwidth=D)
    for mm in gt.lattice.square.dos_moment_coefficients:
        moment = fp.quad(lambda eps: eps**mm * dos(eps), [-D, 0, D])
        assert moment == pytest.approx(gt.square_dos_moment(mm, half_bandwidth=D))


@given(eps=st.floats(-1.5, +1.5))
def test_square_dos_vs_dos_mp(eps):
    """Compare multi-precision and `numpy` implementation of DOS."""
    D = 1.3
    assert np.allclose(gt.square_dos(eps, half_bandwidth=D),
                       float(gt.lattice.square.dos_mp(eps, half_bandwidth=D)))


@pytest.mark.parametrize("gamma", [1.5, 2.])
@pytest.mark.parametrize("D", [0.5, 1., 2.])
def test_rectangular_dos_unit(D, gamma):
    """Integral over the whole DOS should be 1."""
    dos = partial(gt.lattice.rectangular.dos, half_bandwidth=D, scale=gamma)
    singularity = D * (gamma - 1) / (gamma + 1)
    assert fp.quad(dos, [-D, -singularity, singularity, D]) == pytest.approx(1.)


@pytest.mark.parametrize("gamma", [1.5, 2.])
@pytest.mark.parametrize("D", [0.5, 1., 2.])
def test_rectangular_dos_half(D, gamma):
    """DOS should be symmetric -> integral over the half should yield 0.5."""
    dos = partial(gt.lattice.rectangular.dos, half_bandwidth=D, scale=gamma)
    singularity = D * (gamma - 1) / (gamma + 1)
    assert fp.quad(dos, [-D, -singularity, 0.]) == pytest.approx(.5)
    assert fp.quad(dos, [0., +singularity, +D]) == pytest.approx(.5)


@pytest.mark.parametrize("gamma", [1.5, 2.])
def test_rectangular_dos_support(gamma):
    """DOS should have no support for | eps | > D."""
    D = 1.2
    for eps in np.linspace(D + 1e-6, D*1e4):
        assert gt.lattice.rectangular.dos(eps, D, scale=gamma) == 0
        assert gt.lattice.rectangular.dos(-eps, D, scale=gamma) == 0


@given(z=st.complex_numbers(allow_infinity=False, max_magnitude=1e8))
def test_rectangular_vs_square_gf(z):
    """For `scale=1` the rectangular equals the square lattice."""
    assume(abs(z.imag) > 1e-6)
    assert gt.lattice.rectangular.gf_z(z, 1, 1) == pytest.approx(gt.square_gf_z(z, 1))


@pytest.mark.parametrize("D", [0.5, 1., 2.])
def test_triangular_dos_unit(D):
    """Integral over the whole DOS should be 1."""
    dos = partial(gt.lattice.triangular.dos, half_bandwidth=D)
    assert fp.quad(dos, [-2*D/3, -4*D/9, 4*D/3]) == pytest.approx(1.)


def test_triangular_dos_support():
    """DOS should have no support for | eps | > D."""
    D = 1.2
    left = -2*D/3
    right = 4*D/3
    for eps in np.linspace(right + 1e-6, right*1e4):
        assert gt.lattice.triangular.dos(eps, D) == 0
    for eps in np.linspace(left*1e4, left - 1e-6):
        assert gt.lattice.triangular.dos(eps, D) == 0


def test_triangular_imag_gf_negative():
    """Imaginary part of Gf must be smaller or equal 0 for real frequencies."""
    D = 1.2
    omega, omega_step = np.linspace(-D, D, dtype=np.complex, retstep=True)
    omega += 5j*omega_step
    assert np.all(gt.lattice.triangular.gf_z(omega, D).imag <= 0)


def test_triangular_imag_gf_equals_dos():
    r"""Imaginary part of the GF is proportional to the DOS.

    .. math:: DOS(ϵ) = -ℑ(G(ϵ))/π

    """
    D = 1.2
    num = int(1e3)
    # lower band-edge 2/3*D is problematic
    omega = np.linspace(-2/3*D+1e-6, 2*D, dtype=np.complex, num=num)
    omega += 1e-16j
    assert np.allclose(-gt.lattice.triangular.gf_z(omega, D).imag/np.pi,
                       gt.lattice.triangular.dos(omega.real, D))


@pytest.mark.parametrize("D", [0.5, 1.7, 2.])
def test_triangular_dos_moment(D):
    """Moment is integral over ϵ^m DOS."""
    # check influence of bandwidth, as they are calculated for D=1 and normalized
    dos = partial(gt.lattice.triangular.dos, half_bandwidth=D)
    for mm in gt.lattice.triangular.dos_moment_coefficients:
        # fp.quad fails for some values of D
        # moment = fp.quad(lambda eps: eps**mm * dos(eps), [-2/3*D, -4/9*D, 4/3*D])
        moment, __ = integrate.quad(lambda eps: eps**mm * dos(eps),
                                    -2/3*D, 4/3*D, points=[-4/9*D])
        assert moment == pytest.approx(gt.lattice.triangular.dos_moment(mm, half_bandwidth=D))


@given(eps=st.floats(-1.5, +1.5))
def test_triangular_dos_vs_dos_mp(eps):
    """Compare multi-precision and `numpy` implementation of DOS."""
    D = 1.3
    assert np.allclose(gt.lattice.triangular.dos(eps, half_bandwidth=D),
                       float(gt.lattice.triangular.dos_mp(eps, half_bandwidth=D)))


@pytest.mark.parametrize("D", [0.5, 1., 2.])
def test_honeycomb_dos_unit(D):
    """Integral over the whole DOS should be 1."""
    dos = partial(gt.lattice.honeycomb.dos, half_bandwidth=D)
    assert fp.quad(dos, [-D, -D/3, 0, D/3, D]) == pytest.approx(1.)
    assert integrate.quad(dos, -D, D, points=[-D/3, 0, D/3])[0] == pytest.approx(1.)


def test_honeycomb_dos_support():
    """DOS should have no support for | eps | > D."""
    D = 1.2
    for eps in np.linspace(D + 1e-6, D*1e4):
        assert gt.lattice.honeycomb.dos(+eps, D) == 0
        assert gt.lattice.honeycomb.dos(-eps, D) == 0


def test_honeycomb_imag_gf_negative():
    """Imaginary part of Gf must be smaller or equal 0 for real frequencies."""
    D = 1.2
    omega, omega_step = np.linspace(-D, D, dtype=np.complex, retstep=True)
    omega += 5j*omega_step
    assert np.all(gt.lattice.honeycomb.gf_z(omega, D).imag <= 0)


def test_honeycomb_imag_gf_equals_dos():
    r"""Imaginary part of the GF is proportional to the DOS.

    .. math:: DOS(ϵ) = -ℑ(G(ϵ))/π

    """
    D = 1.2
    # around band-edge, there are some issues, cmp. triangular lattice
    delta = 1e-4
    omega = np.linspace(-D+delta, +D+delta, num=int(1e3)) + 1e-16j
    assert np.allclose(-gt.lattice.honeycomb.gf_z(omega, D).imag/np.pi,
                       gt.lattice.honeycomb.dos(omega.real, D))


@pytest.mark.parametrize("D", [0.5, 1., 2.])
def test_kagome_dos_unit(D):
    """Integral over the whole DOS should be 2/3, delta-peak is excluded."""
    dos = partial(gt.lattice.kagome.dos, half_bandwidth=D)
    assert fp.quad(dos, [-2*D/3, 0, D/3, 2*D/3, 4*D/3]) == pytest.approx(2/3)
    assert integrate.quad(dos, -2*D/3, 4*D/3, points=[0, D/3, 2*D/3])[0] == pytest.approx(2/3)


def test_kagome_dos_support():
    """DOS should have no support for | eps | > D."""
    D = 1.2
    left = -2*D/3
    right = 4*D/3
    for eps in np.linspace(right + 1e-6, right*1e4):
        assert gt.lattice.kagome.dos(eps, D) == 0
    for eps in np.linspace(left*1e4, left - 1e-6):
        assert gt.lattice.kagome.dos(eps, D) == 0


def test_kagome_imag_gf_negative():
    """Imaginary part of Gf must be smaller or equal 0 for real frequencies."""
    D = 1.2
    omega, omega_step = np.linspace(-D, D, dtype=np.complex, retstep=True)
    omega += 5j*omega_step
    assert np.all(gt.lattice.kagome.gf_z(omega, D).imag <= 0)


def test_kagome_imag_gf_equals_dos():
    r"""Imaginary part of the GF is proportional to the DOS away from delta.

    .. math:: DOS(ϵ) = -ℑ(G(ϵ))/π

    """
    D = 1.2
    # around band-edge, there are some issues, cmp. triangular lattice
    delta = 1e-3
    omega = np.linspace(-D+delta, +D+delta, num=int(1e3)) + 1e-16j
    assert np.allclose(-gt.lattice.kagome.gf_z(omega, D).imag/np.pi,
                       gt.lattice.kagome.dos(omega.real, D))


@pytest.mark.parametrize("D", [0.5, 1.7, 2.])
def test_simplecubic_dos_moment(D):
    """Moment is integral over ϵ^m DOS."""
    # check influence of bandwidth, as they are calculated for D=1 and normalized
    dos = partial(gt.lattice.honeycomb.dos, half_bandwidth=D)
    for mm in gt.lattice.honeycomb.dos_moment_coefficients:
        moment = fp.quad(lambda eps: eps**mm * dos(eps), [-D, -D/3, 0, +D/3, +D])
        assert moment == pytest.approx(gt.lattice.honeycomb.dos_moment(mm, half_bandwidth=D))


@given(eps=st.floats(-1.5, +1.5))
def test_honeycomb_dos_vs_dos_mp(eps):
    """Compare multi-precision and numpy implementation of GF."""
    D = 1.3
    assert np.allclose(gt.lattice.honeycomb.dos(eps, half_bandwidth=D),
                       float(gt.lattice.honeycomb.dos_mp(eps, half_bandwidth=D)))


@pytest.mark.parametrize("D", [0.5, 1., 2.])
def test_simplecubic_dos_unit(D):
    """Integral over the whole DOS should be 1."""
    dos = partial(gt.sc_dos, half_bandwidth=D)
    assert fp.quad(dos, [-D, -D/3, 0, D/3, D]) == pytest.approx(1.)


@pytest.mark.parametrize("D", [0.5, 1., 2.])
def test_simplecubic_dos_half(D):
    """DOS should be symmetric -> integral over the half should yield 0.5."""
    dos = partial(gt.sc_dos, half_bandwidth=D)
    assert fp.quad(dos, [-D, -D/3, 0]) == pytest.approx(0.5)
    assert fp.quad(dos, [0, +D/3, +D]) == pytest.approx(0.5)


def test_simplecubic_dos_support():
    """DOS should have no support for | eps | > D."""
    D = 1.2
    for eps in np.linspace(D + 1e-6, D*1e4):
        assert gt.sc_dos(+eps, D) == 0
        assert gt.sc_dos(-eps, D) == 0


def test_simplecubic_imag_gf_negative():
    """Imaginary part of Gf must be smaller or equal 0 for real frequencies."""
    D = 1.2
    omega, omega_step = np.linspace(-D, D, dtype=np.complex, retstep=True)
    omega += 5j*omega_step
    assert np.all(gt.sc_gf_z(omega, D).imag <= 0)


def test_simplecubic_imag_gf_equals_dos():
    r"""Imaginary part of the GF is proportional to the DOS.

    .. math:: DOS(ϵ) = -ℑ(G(ϵ))/π

    """
    D = 1.2
    num = int(1e3)
    omega = np.linspace(-D, D, num=num) + 1e-16j
    assert np.allclose(-gt.sc_gf_z(omega, D).imag/np.pi,
                       gt.sc_dos(omega.real, D))


@pytest.mark.parametrize("D", [0.5, 1.7, 2.])
def test_simplecubic_dos_moment(D):
    """Moment is integral over ϵ^m DOS."""
    # check influence of bandwidth, as they are calculated for D=1 and normalized
    dos = partial(gt.lattice.sc.dos, half_bandwidth=D)
    for mm in gt.lattice.sc.dos_moment_coefficients:
        moment = fp.quad(lambda eps: eps**mm * dos(eps), [-D, -D/3, D/3, D])
        assert moment == pytest.approx(gt.sc_dos_moment(mm, half_bandwidth=D))


@pytest.mark.parametrize("D", [0.5, 1., 2.])
@given(z=st.complex_numbers(max_magnitude=1e6))
def test_simplecubic_gf_vs_gf_mp(z, D):
    """Compare multi-precision and numpy implementation of GF."""
    assume(abs(z.imag) > 1e-6)
    assert np.allclose(gt.sc_gf_z(z, half_bandwidth=D),
                       complex(gt.lattice.sc.gf_z_mp(z, half_bandwidth=D)))


@given(eps=st.floats(-1.5, +1.5))
def test_simplecubic_dos_vs_dos_mp(eps):
    """Compare multi-precision and numpy implementation of GF."""
    D = 1.3
    assert np.allclose(gt.sc_dos(eps, half_bandwidth=D),
                       float(gt.lattice.sc.dos_mp(eps, half_bandwidth=D)))


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
