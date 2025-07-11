"""Test Bethe lattice for general coordination number.

We compare it to the limits `gftool.lattice.bethe` and `gftool.lattice.onedim`.

"""
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import gftool as gt

bethe = gt.lattice.bethe
bethez = gt.lattice.bethez
onedim = gt.lattice.onedim


@given(z=st.complex_numbers(max_magnitude=1e4).filter(lambda z: abs(z.imag) > 1e-16),
       half_bandwidth=st.floats(min_value=0.1, max_value=2))
def test_inifinite_coordination(z, half_bandwidth):
    """Compare `bethez` for large coordination with `bethe`."""
    coordination = int(1e8)  # huge but finite value
    assert_allclose(
        bethe.gf_z(z, half_bandwidth=half_bandwidth),
        bethez.gf_z(z, half_bandwidth=half_bandwidth, coordination=coordination)
    )


@given(z=st.complex_numbers(max_magnitude=1e4).filter(lambda z: abs(z.imag) > 1e-16),
       half_bandwidth=st.floats(min_value=0.1, max_value=2))
def test_coordination2(z, half_bandwidth):
    """Compare `bethez` for `coordination=2` with `onedim`."""
    assert_allclose(
        onedim.gf_z(z, half_bandwidth=half_bandwidth),
        bethez.gf_z(z, half_bandwidth=half_bandwidth, coordination=2)
    )


@given(eps=st.floats(-1.0, 1.0),
       half_bandwidth=st.floats(min_value=0.1, max_value=2))
def test_inifinite_coordination_dos(eps, half_bandwidth):
    """Compare `bethez` for large coordination with `bethe`."""
    eps *= half_bandwidth
    coordination = int(1e8)  # huge but finite value
    assert_allclose(
        bethe.dos(eps, half_bandwidth=half_bandwidth),
        bethez.dos(eps, half_bandwidth=half_bandwidth, coordination=coordination)
    )


@given(eps=st.floats(-1, 1, exclude_min=True, exclude_max=True),
       half_bandwidth=st.floats(min_value=0.1, max_value=2))
def test_coordination2_dos(eps, half_bandwidth):
    """Compare `bethez` for `coordination=2` with `onedim`."""
    eps *= half_bandwidth
    assert_allclose(
        onedim.dos(eps, half_bandwidth=half_bandwidth),
        bethez.dos(eps, half_bandwidth=half_bandwidth, coordination=2)
    )
