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

    def gf(self, z, **kwds):
        """Green's function."""
        raise NotImplementedError('This is just a placeholder')

    @staticmethod
    @pytest.fixture
    def params():
        """Contains possible parameters needed for the Green's function."""
        return (), {}

    @staticmethod
    def band_edges(params):
        """Return the support of the Green's function, by default (-∞, ∞).

        Can be overwritten by subclasses using the `params`.
        """
        del params
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

    def test_imag_gf_negative(self, params):
        """Imaginary part of Gf must be smaller or equal 0 for real frequencies."""
        omega, omega_step = np.linspace(-10, 10, dtype=complex, retstep=True)
        omega += 5j*omega_step
        assert np.all(self.gf(omega, *params[0], **params[1]).imag <= 0)


class Lattice:
    """Generic class to test basic properties of `gftool.lattice` modules.

    Mostly checks the DOS and it's relation to other functions.

    """

    lattice = gt.lattice.bethe  # should be replaced!! Just available members

    @staticmethod
    @pytest.fixture(scope="class")
    def kwds(request):
        """Contains possible keyword parameters needed for the Green's function."""
        del request
        return {}

    @staticmethod
    def band_edges(**kwds):
        """Return the support of the Green's function, by default (-∞, ∞).

        Can be overwritten by subclasses using the `kwds`.
        """
        del kwds
        return -np.infty, np.infty

    @staticmethod
    def singularities(**kwds):
        """Return singularities for integrations."""
        del kwds
        return []

    def test_dos_unit(self, kwds):
        """Integral over the whole DOS should be 1."""
        dos = partial(self.lattice.dos, **kwds)
        points = self.singularities(**kwds)
        assert integrate.quad(dos, *self.band_edges(**kwds), points=points)[0] == pytest.approx(1.0)

    def test_imgf_eq_dos(self, kwds):
        r"""Imaginary part of the GF is proportional to the DOS.

        .. math:: DOS(ϵ) = -ℑG(ϵ+i0⁺)/π

        """
        omega = np.linspace(*self.band_edges(**kwds), dtype=complex, num=int(1e4)) + 1e-16j
        omega = omega[1:-1]  # exclude endpoints...
        notsingular = np.all(abs(omega.real[:, None] - self.singularities(**kwds)) > 1e-6, axis=-1)
        omega = omega[notsingular]  # compare only away from singularities
        assert np.allclose(-1/np.pi*self.lattice.gf_z(omega, **kwds).imag,
                           self.lattice.dos(omega.real, **kwds), atol=1e-7)

    def test_dos_moment(self, kwds):
        """Moment is integral over ϵ^m DOS."""
        # check influence of bandwidth, as they are calculated for D=1 and normalized
        dos = partial(self.lattice.dos, **kwds)
        dos_moment = partial(self.lattice.dos_moment, **kwds)
        left, right = self.band_edges(**kwds)
        points = [left, *self.singularities(**kwds), right]
        for mm in self.lattice.dos_moment_coefficients:
            # pytint: disable=cell-var-from-loop
            moment = fp.quad(lambda eps: eps**mm * dos(eps), points)
            assert moment == pytest.approx(dos_moment(mm))

    @given(eps=st.floats(-1.5, +1.5))
    def test_dos_vs_dos_mp(self, eps, kwds):
        """Compare multi-precision and `numpy` implementation of DOS."""
        assert np.allclose(self.lattice.dos(eps, **kwds),
                           float(self.lattice.dos_mp(eps, **kwds)))

    def test_dos_support(self, kwds):
        """DOS should have no support for outside the band-edges."""
        lower, upper = self.band_edges(**kwds)
        assert upper > 0, "Else this test-case is ill-defined"
        for eps in np.linspace(upper + 1e-6, upper*1e4):
            assert self.lattice.dos(eps, **kwds) == 0
        assert lower < 0, "Else this test-case is ill-defined"
        for eps in np.linspace(lower - 1e-6, lower*1e4):
            assert self.lattice.dos(eps, **kwds) == 0

    @given(z=st.complex_numbers(max_magnitude=1e6))
    def test_hilbert_transform(self, z, kwds):
        """Hilbert transform is same as non-interacting local Green's function.

        Probably we should drop the Hilbert transform to avoid redundancy,
        but as long as we have let's make sure it also is correct.
        """
        assume(abs(z.imag) > 1e-6)  # avoid singularities on real axis
        assert (self.lattice.hilbert_transform(z, **kwds)
                == pytest.approx(self.lattice.gf_z(z, **kwds)))


class SymLattice(Lattice):
    """Generic class to test basic properties of symmetric `gftool.lattice` modules.

    Mostly checks the DOS and it's relation to other functions.

    """

    def test_dos_half(self, kwds):
        """DOS should be symmetric -> integral over the half should yield 0.5."""
        dos = partial(self.lattice.dos, **kwds)
        mD, D = self.band_edges(**kwds)
        assert mD == -D
        points = self.singularities(**kwds)
        if points:
            assert fp.quad(dos, [-D, *points[:(len(points)+1)//2], 0]) == pytest.approx(0.5)
            assert fp.quad(dos, [0, *points[len(points)//2:], +D]) == pytest.approx(0.5)
        else:
            assert fp.quad(dos, [-D, 0.]) == pytest.approx(.5)
            assert fp.quad(dos, [0., +D]) == pytest.approx(.5)

    @given(mm=st.integers(min_value=0))
    def test_odd_dos_moments(self, mm, kwds):
        """Odd moments vanish for even DOS."""
        m_odd = 2*mm + 1
        assert self.lattice.dos_moment(m_odd, **kwds) == 0.


class TestBetheGf(GfProperties):
    """Check properties of Bethe Gf."""

    D = 1.2
    z_mesh = np.mgrid[-2*D:2*D:5j, -2*D:2*D:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gt.bethe_gf_z)

    @staticmethod
    @pytest.fixture(params=[0.7, 1.2, ])
    def params(request):
        """Parameters for Bethe Green's function."""
        return (), {'half_bandwidth': request.param}


class TestBethe(SymLattice):
    """Check basic properties of `gftool.bethe.lattice`."""

    lattice = gt.lattice.bethe

    @pytest.fixture(params=[0.5, 1., 2.], scope="class")
    def kwds(self, request):
        """Half-bandwidth of Bethe lattice."""
        return {"half_bandwidth": request.param}

    @staticmethod
    def band_edges(half_bandwidth):
        """Return band-edges."""
        return -half_bandwidth, half_bandwidth


class TestOnedimGf(GfProperties):
    """Check properties of one-dimensional Gf."""

    D = 1.2
    z_mesh = np.mgrid[-2*D:2*D:5j, -2*D:2*D:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gt.onedim_gf_z)

    @staticmethod
    @pytest.fixture(params=[0.7, 1.2])
    def params(request):
        """Parameters for Bethe Green's function."""
        return (), {'half_bandwidth': request.param}


class TestOnedim(SymLattice):
    """Check basic properties of `gftool.lattice.onedim`."""

    lattice = gt.lattice.onedim

    @staticmethod
    @pytest.fixture(params=[0.5, 1., 2.], scope="class")
    def kwds(request):
        """Half-bandwidth of Onedim lattice."""
        return {"half_bandwidth": request.param}

    @staticmethod
    def band_edges(half_bandwidth):
        """Return band-edges."""
        return -half_bandwidth, half_bandwidth


class TestSquareGf(GfProperties):
    """Check properties of square Gf."""

    D = 1.2
    z_mesh = np.mgrid[-2*D:2*D:5j, -2*D:2*D:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gt.square_gf_z)

    @staticmethod
    @pytest.fixture(params=[0.7, 1.2, ])
    def params(request):
        """Parameters for Bethe Green's function."""
        return (), {'half_bandwidth': request.param}


class TestSquare(SymLattice):
    """Check basic properties of `gftool.bethe.lattice`."""

    lattice = gt.lattice.square

    @staticmethod
    @pytest.fixture(params=[0.5, 1., 2.], scope="class")
    def kwds(request):
        """Half-bandwidth of square lattice."""
        return {"half_bandwidth": request.param}

    @staticmethod
    def band_edges(half_bandwidth):
        """Return band-edges."""
        return -half_bandwidth, half_bandwidth

    @staticmethod
    def singularities(half_bandwidth):
        """Return singularities."""
        del half_bandwidth
        return [0]


class TestRectangularGf(GfProperties):
    """Check properties of rectangular Gf."""

    D = 1.2
    z_mesh = np.mgrid[-2*D:2*D:5j, -2*D:2*D:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gt.lattice.rectangular.gf_z)

    @staticmethod
    @pytest.fixture(params=[{'half_bandwidth': D, 'scale': gamma}
                            for D, gamma in product([0.7, 1.2], [1.3, 2.0])])
    def params(request):
        """Parameters for Bethe Green's function."""
        return (), request.param

    @staticmethod
    def band_edges(params):
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

    @staticmethod
    @pytest.fixture(params=[0.7, 1.2, ])
    def params(request):
        """Parameters for Bethe Green's function."""
        return (), {'half_bandwidth': request.param}

    @staticmethod
    def band_edges(params):
        """Return the support of the Green's function."""
        D = params[1]['half_bandwidth']
        return -2*D/3, 4*D/3

    def test_normalization(self, params, points=None):
        """Singularities are needed for accurate integration."""
        del points  # was only give for subclasses
        D = params[1]['half_bandwidth']
        super().test_normalization(params, points=[-4*D/9])


class TestTriangular(Lattice):
    """Check basic properties of `gftool.lattice.triangular`."""

    lattice = gt.lattice.triangular

    @staticmethod
    @pytest.fixture(params=[0.5, 1., 2.], scope="class")
    def kwds(request):
        """Half-bandwidth of triangular lattice."""
        return {"half_bandwidth": request.param}

    @staticmethod
    def band_edges(half_bandwidth):
        """Return band-edges."""
        return -2*half_bandwidth/3, 4*half_bandwidth/3

    @staticmethod
    def singularities(half_bandwidth):
        """Return singularities."""
        return [-4*half_bandwidth/9]


class TestHoneycombGf(GfProperties):
    """Check properties of rectangular Gf."""

    D = 1.2
    z_mesh = np.mgrid[-2*D:2*D:5j, -2*D:2*D:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gt.lattice.honeycomb.gf_z)

    @staticmethod
    @pytest.fixture(params=[0.7, 1.2, ])
    def params(request):
        """Parameters for honeycomb Green's function."""
        return (), {'half_bandwidth': request.param}

    @staticmethod
    def band_edges(params):
        """Return the support of the Green's function."""
        D = params[1]['half_bandwidth']
        return -D, D

    def test_normalization(self, params, points=None):
        """Singularities are needed for accurate integration."""
        del points  # was only give for subclasses
        D = params[1]['half_bandwidth']
        super().test_normalization(params, points=[-D/3, +D/3])


class TestHoneycomb(SymLattice):
    """Check basic properties of `gftool.lattice.honeycomb`."""

    lattice = gt.lattice.honeycomb

    @staticmethod
    @pytest.fixture(params=[0.5, 1., 2.], scope="class")
    def kwds(request):
        """Half-bandwidth of honeycomb lattice."""
        return {"half_bandwidth": request.param}

    @staticmethod
    def band_edges(half_bandwidth):
        """Return band-edges."""
        return -half_bandwidth, half_bandwidth

    @staticmethod
    def singularities(half_bandwidth):
        """Return singularities."""
        return [-half_bandwidth/3, half_bandwidth/3]


class TestKagomeGf(GfProperties):
    """Check properties of rectangular Gf."""

    D = 1.2
    z_mesh = np.mgrid[-2*D:2*D:5j, -2*D:2*D:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gt.lattice.kagome.gf_z)

    @staticmethod
    @pytest.fixture(params=[0.7, 1.2, ])
    def params(request):
        """Parameters for kagome Green's function."""
        return (), {'half_bandwidth': request.param}

    @staticmethod
    def band_edges(params):
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


class TestKagome(Lattice):
    """Check basic properties of `gftool.lattice.kagome`."""

    lattice = gt.lattice.kagome

    @staticmethod
    @pytest.fixture(params=[0.5, 1., 2.], scope="class")
    def kwds(request):
        """Half-bandwidth of kagome lattice."""
        return {"half_bandwidth": request.param}

    @staticmethod
    def band_edges(half_bandwidth):
        """Return band-edges."""
        return -2*half_bandwidth/3, 4*half_bandwidth/3

    @staticmethod
    def singularities(half_bandwidth):
        """Return singularities."""
        return [0, half_bandwidth/3, 2*half_bandwidth/3, 2*half_bandwidth/3]

    def test_dos_unit(self, kwds):
        """Integral over the whole DOS should be 2/3, delta-peak is excluded."""
        dos = partial(self.lattice.dos, **kwds)
        points = self.singularities(**kwds)
        assert integrate.quad(dos, *self.band_edges(**kwds), points=points)[0] == pytest.approx(2/3)

    def test_dos_moment(self, kwds):
        """Moment is integral over ϵ^m DOS."""
        # check influence of bandwidth, as they are calculated for D=1 and normalized
        dos = partial(self.lattice.dos, **kwds)
        dos_moment = partial(self.lattice.dos_moment, **kwds)
        left, right = self.band_edges(**kwds)
        points = [left, *self.singularities(**kwds), right]
        D = kwds["half_bandwidth"]
        for mm in self.lattice.dos_moment_coefficients:
            # pytint: disable=cell-var-from-loop
            moment = fp.quad(lambda eps: eps**mm * dos(eps), points)
            moment += (-2*D/3)**mm / 3  # add delta peak by hand
            assert moment == pytest.approx(dos_moment(mm))


class TestLiebGf(GfProperties):
    """Check properties of rectangular Gf."""

    D = 1.2
    z_mesh = np.mgrid[-2*D:2*D:5j, -2*D:2*D:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gt.lattice.lieb.gf_z)

    @staticmethod
    @pytest.fixture(params=[0.7, 1.2, ])
    def params(request):
        """Parameters for Lieb Green's function."""
        return (), {'half_bandwidth': request.param}

    @staticmethod
    def band_edges(params):
        """Return the support of the Green's function."""
        D = params[1]['half_bandwidth']
        return -D, D

    @pytest.mark.filterwarnings("ignore::UserWarning")  # ignores quad's IntegrationWarning
    def test_normalization(self, params, points=None):
        """Due to delta-peak we get only small accuracy."""
        D = params[1]['half_bandwidth']

        def dos(omega):
            r"""Wrap the DOS :math:`ρ(ω) = -ℑG(ω+iϵ)/π`."""
            omega = omega + 1e-6j
            return -self.gf(omega, *params[0], **params[1]).imag/np.pi

        points = np.array([-2**-0.5, -1e-4, 0, +1e-4, +2**-0.5]) * D
        assert (integrate.quad(dos, a=-D, b=+D, points=points)[0]
                == pytest.approx(1, rel=1e-3))


class TestLieb(Lattice):
    """Check basic properties of `gftool.lattice.lieb`."""

    lattice = gt.lattice.lieb

    @staticmethod
    @pytest.fixture(params=[0.5, 1., 2.], scope="class")
    def kwds(request):
        """Half-bandwidth of lieb lattice."""
        return {"half_bandwidth": request.param}

    @staticmethod
    def band_edges(half_bandwidth):
        """Return band-edges."""
        return -half_bandwidth, half_bandwidth

    @staticmethod
    def singularities(half_bandwidth):
        """Return singularities."""
        singular = half_bandwidth * 2**-0.5
        return [-singular, +singular]

    def test_dos_unit(self, kwds):
        """Integral over the whole DOS should be 2/3, delta-peak is excluded."""
        dos = partial(self.lattice.dos, **kwds)
        points = self.singularities(**kwds)
        assert integrate.quad(dos, *self.band_edges(**kwds), points=points)[0] == pytest.approx(2/3)

    def test_dos_moment(self, kwds):
        """Moment is integral over ϵ^m DOS."""
        # check influence of bandwidth, as they are calculated for D=1 and normalized
        dos = partial(self.lattice.dos, **kwds)
        dos_moment = partial(self.lattice.dos_moment, **kwds)
        left, right = self.band_edges(**kwds)
        points = [left, *self.singularities(**kwds), right]
        for mm in self.lattice.dos_moment_coefficients:
            # pytint: disable=cell-var-from-loop
            if kwds["half_bandwidth"] < 0.8 and mm > 10:
                break  # small integrals are numerically inaccurate
            # moment = fp.quad(lambda eps: eps**mm * dos(eps), points)
            # fp.quad fails for some values of D
            # moment = fp.quad(lambda eps: eps**mm * dos(eps), interval)
            moment, __ = integrate.quad(lambda eps: eps**mm * dos(eps),
                                        points[0], points[-1], points=points[1:-1])
            if mm == 0:  # delta peak contributes
                moment += 1/3
            assert moment == pytest.approx(dos_moment(mm))


class TestSimpleCubicGf(GfProperties):
    """Check properties of rectangular Gf."""

    D = 1.2
    z_mesh = np.mgrid[-2*D:2*D:5j, -2*D:2*D:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gt.lattice.sc.gf_z)

    @staticmethod
    @pytest.fixture(params=[0.7, 1.2, ])
    def params(request):
        """Parameters for simple cubic Green's function."""
        return (), {'half_bandwidth': request.param}

    @staticmethod
    def band_edges(params):
        """Return the support of the Green's function."""
        D = params[1]['half_bandwidth']
        return -D, D

    def test_normalization(self, params, points=None):
        """Singularities are needed for accurate integration."""
        del points  # was only give for subclasses
        D = params[1]['half_bandwidth']
        super().test_normalization(params, points=[-D/3, D/3])


class TestSimpleCubic(SymLattice):
    """Check basic properties of `gftool.lattice.sc`."""

    lattice = gt.lattice.sc

    @staticmethod
    @pytest.fixture(params=[0.5, 1., 2.], scope="class")
    def kwds(request):
        """Half-bandwidth of simple cubic lattice."""
        return {"half_bandwidth": request.param}

    @staticmethod
    def band_edges(half_bandwidth):
        """Return band-edges."""
        return -half_bandwidth, half_bandwidth

    @staticmethod
    def singularities(half_bandwidth):
        """Return singularities."""
        return [-half_bandwidth/3, half_bandwidth/3]


class TestBodyCenteredCubicGf(GfProperties):
    """Check properties of bcc Gf."""

    D = 1.2
    z_mesh = np.mgrid[-2*D:2*D:5j, -2*D:2*D:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gt.lattice.bcc.gf_z)

    @staticmethod
    @pytest.fixture(params=[0.7, 1.2, ])
    def params(request):
        """Parameters for simple cubic Green's function."""
        return (), {'half_bandwidth': request.param}

    @staticmethod
    def band_edges(params):
        """Return the support of the Green's function."""
        D = params[1]['half_bandwidth']
        return -D, D

    def test_normalization(self, params, points=None):
        """Singularities are needed for accurate integration."""
        del points  # was only give for subclasses
        super().test_normalization(params, points=[0])


class TestBodyCenteredCubic(SymLattice):
    """Check basic properties of `gftool.lattice.bcc`."""

    lattice = gt.lattice.bcc

    @staticmethod
    @pytest.fixture(params=[0.5, 1., 2.], scope="class")
    def kwds(request):
        """Half-bandwidth of simple cubic lattice."""
        return {"half_bandwidth": request.param}

    @staticmethod
    def band_edges(half_bandwidth):
        """Return band-edges."""
        return -half_bandwidth, half_bandwidth

    @staticmethod
    def singularities(half_bandwidth):
        """Return singularities."""
        del half_bandwidth
        return [0]


class TestFaceCenteredCubicGf(GfProperties):
    """Check properties of fcc Gf."""

    D = 1.2
    z_mesh = np.mgrid[-2*D:2*D:5j, -2*D:2*D:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gt.lattice.fcc.gf_z)

    @staticmethod
    @pytest.fixture(params=[0.7, 1.2, ])
    def params(request):
        """Parameters for simple cubic Green's function."""
        return (), {'half_bandwidth': request.param}

    @staticmethod
    def band_edges(params):
        """Return the support of the Green's function."""
        D = params[1]['half_bandwidth']
        return -0.5*D, 1.5*D

    def test_normalization(self, params, points=None):
        """Singularities are needed for accurate integration."""
        del points  # was only give for subclasses
        super().test_normalization(params, points=[0])


class TestFaceCenteredCubic(Lattice):
    """Check basic properties of `gftool.lattice.fcc`."""

    lattice = gt.lattice.fcc

    @staticmethod
    @pytest.fixture(params=[0.5, 1., 2.], scope="class")
    def kwds(request):
        """Half-bandwidth of simple cubic lattice."""
        return {"half_bandwidth": request.param}

    @staticmethod
    def band_edges(half_bandwidth):
        """Return band-edges."""
        return -0.5*half_bandwidth, 1.5*half_bandwidth

    @staticmethod
    def singularities(half_bandwidth):
        """Return singularities."""
        del half_bandwidth
        return [0]

    def test_dos_moment(self, kwds):
        """Moment is integral over ϵ^m DOS.

        Overwritten to soften the absolute tolerance for ``m1`` which 0.
        """
        # check influence of bandwidth, as they are calculated for D=1 and normalized
        dos = partial(self.lattice.dos, **kwds)
        dos_moment = partial(self.lattice.dos_moment, **kwds)
        left, right = self.band_edges(**kwds)
        points = [left, *self.singularities(**kwds), right]
        for mm in self.lattice.dos_moment_coefficients:
            # pytint: disable=cell-var-from-loop
            moment = fp.quad(lambda eps: eps**mm * dos(eps), points)
            assert moment == pytest.approx(dos_moment(mm), rel=1e-6, abs=1e-10)


class TestSurfaceGf(GfProperties):
    """Check properties of surface Gf."""

    z_mesh = np.mgrid[-2:2:5j, -2:2:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gt.surface_gf_zeps)

    @staticmethod
    @pytest.fixture(params=[-.8, -.4, 0., .5, .7])
    def params(request):
        """Parameters for the Surface Bethe Green's function."""
        return (), {'eps': request.param,
                    'hopping_nn': .2,
                    }

    @staticmethod
    def band_edges(params):
        """Bandages are shifted ones of `gt.bethe_gf_z`."""
        hopping_nn = params[1]['hopping_nn']
        eps = params[1]['eps']
        return -2*hopping_nn-abs(eps), 2*hopping_nn+abs(eps)


class TestHubbardDimer(GfProperties):
    """Check properties of Hubbard Dimer Gf."""

    z_mesh = np.mgrid[-2:2:5j, -2:2:4j]
    z_mesh = np.ravel(z_mesh[0] + 1j*z_mesh[1])

    gf = method(gt.hubbard_dimer_gf_z)

    @staticmethod
    @pytest.fixture(params=['+', '-'])
    def params(request):
        """Parameters for the Hubbard Dimer Green's function."""
        return (), {'kind': request.param,
                    'hopping': 1.1,
                    'interaction': 1.3,
                    }

    @pytest.mark.skip(reason="Fixing integral: nearly Delta-functions, no band_edges!")
    def test_normalization(self, params):
        """Atomic functions cannot be integrated numerically."""
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


@given(z=st.complex_numbers(max_magnitude=1e4),
       D=st.floats(min_value=1e-2, max_value=1e2))
def test_bethe_inverse(z, D):
    """Check inverse."""
    assume(z.imag != 0)  # Gf have poles on real axis
    gf = gt.lattice.bethe.gf_z(z, half_bandwidth=D)
    assert gt.lattice.bethe.gf_z_inv(gf, half_bandwidth=D) == pytest.approx(z)


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
@given(z=st.complex_numbers(max_magnitude=1e6))
def test_simplecubic_gf_vs_gf_mp(z, D):
    """Compare multi-precision and numpy implementation of GF."""
    assume(abs(z.imag) > 1e-6)
    assert np.allclose(gt.sc_gf_z(z, half_bandwidth=D),
                       complex(gt.lattice.sc.gf_z_mp(z, half_bandwidth=D)))


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


def test_hubbard_I_self_hfm():
    """Check high-frequency moments of Hubbard-I self-energy."""
    U, occ = 1.7, 0.3
    hubbard_I = partial(gt.hubbard_I_self_z, U=U, occ=occ)
    m0 = U*occ
    limit_0p = fp.limit(hubbard_I, np.infty, exp=True)
    assert limit_0p == pytest.approx(m0)

    def hubbard_dyn(z):
        return z*(hubbard_I(z) - m0)

    m1 = occ * (1 - occ) * U**2
    limit_1p = fp.limit(hubbard_dyn, np.infty, exp=True, steps=[25])
    assert limit_1p == pytest.approx(m1)
