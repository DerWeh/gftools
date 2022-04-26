"""Test PolePadé algorithm."""
from functools import partial

import numpy as np

from .context import gftool as gt

assert_allclose = np.testing.assert_allclose


def test_bethe_unitcircle():
    """Simple PolePadé test to reproduce Bethe DOS under optimal conditions."""
    g0 = partial(gt.bethe_gf_z, half_bandwidth=1)
    # choose optimal sample points: the unit-(half-)circle
    z = np.exp(1j*np.linspace(np.pi, 0, num=252)[1:-1])
    gf_z = g0(z)
    pade = gt.polepade.continuation(z, gf_z)

    # check on the real axis
    # at the band-edge, it will obviously be hard, so we exclude it
    ww = np.linspace(-0.9, 0.9, num=500)
    gf_ww = g0(ww+1e-16j)
    pade_ww = pade.eval_polefct(ww)
    assert_allclose(gf_ww, pade_ww, rtol=1e-4)
    pade_ww = pade.eval_zeropole(ww)
    assert_allclose(gf_ww, pade_ww, rtol=1e-4)

    # check on the imaginary axis
    iws = gt.matsubara_frequencies(range(1024), beta=100)
    gf_iw = g0(iws)
    pade_iw = pade.eval_polefct(iws)
    assert_allclose(gf_iw, pade_iw, rtol=1e-6)
    pade_iw = pade.eval_zeropole(iws)
    assert_allclose(gf_iw, pade_iw, rtol=1e-6)


def test_bethe_iw():
    """Simple PolePadé test to reproduce Bethe DOS from the imaginary axis."""
    g0 = partial(gt.bethe_gf_z, half_bandwidth=1)
    # choose Matsubara axis, large beta for good results
    z = gt.matsubara_frequencies(range(1024), beta=500)
    gf_z = g0(z)
    pade = gt.polepade.continuation(z, gf_z, weight=1/abs(z), moments=[1.])

    # check on the real axis
    # at the band-edge, it will obviously be hard, so we exclude it
    ww = np.linspace(-0.9, 0.9, num=500)
    gf_ww = g0(ww+1e-16j)
    pade_ww = pade.eval_polefct(ww)
    assert_allclose(gf_ww, pade_ww, rtol=5e-3)
    pade_ww = pade.eval_zeropole(ww)
    assert_allclose(gf_ww, pade_ww, rtol=5e-3)

    # check on the imaginary axis
    iws = gt.matsubara_frequencies(range(1024), beta=100)
    gf_iw = g0(iws)
    pade_iw = pade.eval_polefct(iws)
    assert_allclose(gf_iw, pade_iw)
    pade_iw = pade.eval_zeropole(iws)
    assert_allclose(gf_iw, pade_iw)
