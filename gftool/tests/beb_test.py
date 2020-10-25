"""Test BEB."""
from functools import partial

import pytest
import numpy as np
import hypothesis.strategies as st

from scipy import integrate
from hypothesis import given, assume

from .context import gftool as gt


# test for some random handpicked values as it is expensive
@pytest.mark.parametrize("self_beb_z", (
    (0.3-1j)*np.eye(2, 2),
    np.array([[0.178 - 0.13j, 0.4 + 1j],
              [0.4 + 1j, -1.38 - 0.46j]])
))
def test_gf_loc(self_beb_z):
    """Check local Green's function against integration.

    This is a rather expensive test -> currently no hypothesis.
    """
    z = np.array([0])  # only `np.eye()*z - self_beb_z` is relevant
    # symmetrize matrix
    self_beb_z[1, 0] = self_beb_z[0, 1] = 0.5*(self_beb_z[0, 1] + self_beb_z[1, 0])
    assert np.linalg.cond(self_beb_z) < 1e4   # reasonably well condition matrices only
    t = np.array([[1.0, 0.3],
                  [0.3, 1.2]])
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


@given(z=st.complex_numbers(max_magnitude=1e-6))
def test_selfconsistency(z):
    """Self-consistent Gf is diagonal and equals averaged Gf."""
    # randomly chosen example
    assume(z.imag != 0)
    z = np.array(z.conjugate() if z.imag < 0 else z)
    t = np.array([[1.0, 0.3],
                  [0.3, 1.2]])
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
