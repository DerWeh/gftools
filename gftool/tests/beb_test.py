"""Test BEB."""
from functools import partial

import pytest
import numpy as np

from scipy import integrate

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
    # symmetrise matrix
    self_beb_z[1, 0] = self_beb_z[0, 1] = 0.5*(self_beb_z[0, 1] + self_beb_z[1, 0])
    assert np.linalg.cond(self_beb_z) < 1e4   # reasonably well condition matrices only
    t = np.array([[1.0, 0.3],
                  [0.3, 1.2]])
    D = 1.3
    hilbert = partial(gt.bethe_hilbert_transform, half_bandwidth=D)

    gf_z = gt.beb.gf_loc_z(z, self_beb_z, hopping=t, hilbert_trafo=hilbert, diagonal=False)

    # Gf loc using brute-force integration
    kernel = z[..., np.newaxis, np.newaxis]*np.eye(*t.shape) - self_beb_z

    def dos_gf_eps(eps):
        """k-dependent Gf which has to be integrated."""
        return gt.bethe_dos(eps, half_bandwidth=D)*np.linalg.inv(kernel - t*eps)

    gf_cmp, __ = integrate.quad_vec(dos_gf_eps, -D, D)

    assert np.allclose(gf_z, gf_cmp)
