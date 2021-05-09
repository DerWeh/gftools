"""Test BEB."""
import logging

from functools import partial

import pytest
import numpy as np
import hypothesis.strategies as st

from scipy import integrate
from hypothesis import given, assume

from .context import gftool as gt


# test for some random handpicked values as it is expensive
@pytest.mark.parametrize("t", [
    np.array([[1.0, 0.3], [0.3, 1.2]]),  # full rank example
    np.array([[4.0, 6.0], [6.0, 9.0]]),  # rank deficient example
])
@pytest.mark.parametrize("self_beb_z", (
    (0.3-1j)*np.eye(2, 2),
    np.array([[0.178 - 0.13j, 0.4 + 1j],
              [0.4 + 1j, -1.38 - 0.46j]])
))
def test_gf_loc(t, self_beb_z):
    """Check local Green's function against integration.

    This is a rather expensive test -> currently no hypothesis.
    """
    z = np.array([0])  # only `np.eye()*z - self_beb_z` is relevant
    # symmetrize matrix
    self_beb_z[1, 0] = self_beb_z[0, 1] = 0.5*(self_beb_z[0, 1] + self_beb_z[1, 0])
    assert np.linalg.cond(self_beb_z) < 1e4   # reasonably well condition matrices only
    D = 1.3
    hilbert = partial(gt.bethe_hilbert_transform, half_bandwidth=D)

    gf_z = gt.beb.gf_loc_z(z, self_beb_z, hopping=t, hilbert_trafo=hilbert, diag=False)

    # Gf loc using brute-force integration
    kernel = z[..., np.newaxis, np.newaxis]*np.eye(*t.shape) - self_beb_z

    def dos_gf_eps(eps):
        """k-dependent Gf which has to be integrated."""
        return gt.bethe_dos(eps, half_bandwidth=D)*np.linalg.inv(kernel - t*eps)

    gf_cmp, __ = integrate.quad_vec(dos_gf_eps, -D, D)

    assert np.allclose(gf_z, gf_cmp)


@pytest.mark.parametrize("t", [
    np.array([[1.0, 0.3], [0.3, 1.2]]),  # full rank example
    np.array([[4.0, 6.0], [6.0, 9.0]]),  # rank deficient example
])
@given(z=st.complex_numbers(max_magnitude=1e-6))
def test_selfconsistency(t, z):
    """Self-consistent Gf is diagonal and equals averaged Gf."""
    # randomly chosen example
    assume(z.imag != 0)
    z = np.array(z.conjugate() if z.imag < 0 else z)
    eps = np.array([-0.137, 0.23])
    c = np.array([0.137, 0.863])
    D = 1.3
    hilbert = partial(gt.bethe_hilbert_transform, half_bandwidth=D)
    self_beb_z = gt.beb.solve_root(np.array(z), eps, concentration=c, hopping=t,
                                   hilbert_trafo=hilbert, restricted=False)
    gf_loc_z = gt.beb.gf_loc_z(z, self_beb_z, hopping=t, hilbert_trafo=hilbert, diag=False)
    assert np.allclose(gf_loc_z[..., 0, 1], 0)
    assert np.allclose(gf_loc_z[..., 1, 0], 0)
    # Gf_loc diagonal -> inverse is 1/diagonal
    gf_avg = c / (1./gt.beb.diagonal(gf_loc_z) + gt.beb.diagonal(self_beb_z) - eps)
    assert np.allclose(gt.beb.diagonal(gf_loc_z), gf_avg)


@given(z=st.complex_numbers(max_magnitude=1e-6))
def test_cpa_limit(z):
    """Compare BEB Gf to CPA Gf in the CPA limit `t=1`."""
    # randomly chosen example
    assume(z.imag != 0)
    z = np.array(z.conjugate() if z.imag < 0 else z)
    eps = np.array([-0.137, 0.23])
    c = np.array([0.137, 0.863])
    D = 1.3
    hilbert = partial(gt.bethe_hilbert_transform, half_bandwidth=D)
    # cpa limit -> t=1
    t = np.array([[1.0, 1.0],
                  [1.0, 1.0]])
    self_beb_z = gt.beb.solve_root(z, eps, concentration=c, hopping=t,
                                   hilbert_trafo=hilbert)
    gf_loc_z = gt.beb.gf_loc_z(z, self_beb_z, hopping=t, hilbert_trafo=hilbert, diag=True)
    self_cpa_z = gt.cpa.solve_root(z, eps, concentration=c, hilbert_trafo=hilbert)
    gf_cpa_z = gt.cpa.gf_cmpt_z(z, self_cpa_z, e_onsite=eps, hilbert_trafo=hilbert)
    assert np.allclose(gf_loc_z, c*gf_cpa_z)
    assert np.allclose(gf_loc_z.sum(axis=-1), hilbert(z - self_cpa_z))


@pytest.mark.filterwarnings("ignore:(invalid value encountered in double_scalars):RuntimeWarning")
def test_resuming():
    """Calculating the BEB effective medium twice should be no issue."""
    # randomly chosen example
    ww = np.linspace(-2, 2, num=100) + 1e-6j
    ww = ww.reshape((5, 20))
    eps = np.array([-0.37, 0.123])
    c = np.array([0.137, 0.863])
    D = 1.2
    hilbert = partial(gt.bethe_hilbert_transform, half_bandwidth=D)
    t = np.array([[0.5, 1.2],
                  [1.2, 1.0]])
    solve_root = partial(gt.beb.solve_root, ww, eps, concentration=c, hopping=t,
                         hilbert_trafo=hilbert, options=dict(fatol=1e-8))
    self_beb_ww = solve_root()
    self_beb_resume = solve_root(self_beb_z0=self_beb_ww)
    assert np.allclose(self_beb_ww, self_beb_resume)
    # too slow
    # self_beb_0 = solve_root(self_beb_z0=self_beb_ww[0, 0])
    # assert np.allclose(self_beb_ww, self_beb_0)


def test_basic_logging(caplog):
    """At least make sure nothing crashes if we log."""
    caplog.set_level(logging.DEBUG, logger="gftool.beb")
    # randomly chosen example
    ww = np.linspace(-2, 2, num=50) + 1e-3j
    eps = np.array([-0.37, 0.123])
    c = np.array([0.137, 0.863])
    D = 1.2
    hilbert = partial(gt.bethe_hilbert_transform, half_bandwidth=D)
    t = np.array([[0.5, 1.2],
                  [1.2, 1.0]])
    gt.beb.solve_root(ww, eps, concentration=c, hopping=t, hilbert_trafo=hilbert)
