"""Test Bethe lattice for general coordination number.

We compare it to the limits `gftool.lattice.bethe` and `gftool.lattice.onedim`.

"""
import numpy as np

from hypothesis import assume, given, strategies as st

from .context import gftool as gt

bethe = gt.lattice.bethe
bethez = gt.lattice.bethez
onedim = gt.lattice.onedim


@given(z=st.complex_numbers(max_magnitude=1e4),
       half_bandwidth=st.floats(min_value=0.1, max_value=2))
def test_inifinite_coordination(z, half_bandwidth):
    """Compare `bethez` for large coordination with `bethe`."""
    assume(z.imag != 0)
    coordination = int(1e8)  # huge but finite value
    assert np.allclose(
        bethe.gf_z(z, half_bandwidth=half_bandwidth),
        bethez.gf_z(z, half_bandwidth=half_bandwidth, coordination=coordination)
    )


@given(z=st.complex_numbers(max_magnitude=1e4),
       half_bandwidth=st.floats(min_value=0.1, max_value=2))
def test_coordination2(z, half_bandwidth):
    """Compare `bethez` for `coordination=2` with `onedim`."""
    assume(z.imag != 0)
    assert np.allclose(
        onedim.gf_z(z, half_bandwidth=half_bandwidth),
        bethez.gf_z(z, half_bandwidth=half_bandwidth, coordination=2)
    )
