"""Test linear prediction."""
import numpy as np
import pytest

from hypothesis import given, assume, strategies as st
from hypothesis_gufunc.gufunc import gufunc_args

from .context import gftool as gt

lp = gt.linearprediction
assert_allclose = np.testing.assert_allclose


def test_pcoeff_burg_implementation():
    """Test the against the model data given in Kay1988."""
    # Test data
    x = np.zeros(32, dtype=complex)
    x[0] = 6.330 - 0.174915j
    x[1] = -1.3353 - 0.03044j
    x[2] = 3.6189 - 0.260459j
    x[3] = 1.8751 - 0.323974j
    x[4] = -1.0856 - 0.136055j
    x[5] = 3.9911 - 0.101864j
    x[6] = -4.10184 + 0.130571j
    x[7] = 1.55399 + 0.0977916j
    x[8] = -2.125 - 0.306485j
    x[9] = -3.2787 - 0.0544436j
    x[10] = 0.241218 + 0.0962379j
    x[11] = -5.74708 + 0.0186908j
    x[12] = -0.0165977 + 0.237493j
    x[13] = -3.2892 - 0.188478j
    x[14] = -1.3122 - 0.120636j
    x[15] = 0.74525 - 0.0679575j
    x[16] = -1.7719 - 0.416229j
    x[17] = 2.5641 - 0.270373j
    x[18] = 0.2132 - 0.232544j
    x[19] = 2.23409 + 0.236383j
    x[20] = 2.2949 + 0.173061j
    x[21] = 1.09186 + 0.140938j
    x[22] = 2.29353 + 0.442044j
    x[23] = 0.695823 + 0.509325j
    x[24] = 0.759858 + 0.417967j
    x[25] = -0.354267 + 0.506891j
    x[26] = -0.594517 + 0.39708j
    x[27] = -1.88618 + 0.649179j
    x[28] = -1.39041 + 0.867086j
    x[29] = -3.06381 + 0.422965j
    x[30] = -2.0433 + 0.0825514j
    x[31] = -2.162 - 0.0933218j
    # Result data
    A = np.zeros(10, dtype=complex)
    A[0] = -0.10964 + 9.80833E-02j
    A[1] = -1.07989 - 8.23723E-02j
    A[2] = -0.16670 - 7.09936E-02j
    A[3] = 0.86892 + 0.14021j
    A[4] = -0.60076 - 3.46992E-02j
    A[5] = 0.34997 - 0.34456j
    A[6] = 0.59898 + 0.15030j
    A[7] = -0.20271 + 0.34466j
    A[8] = -0.50447 - 3.14139E-02j
    A[9] = 0.37516 - 0.18268j
    SIG2 = 0.16930
    cmp_a, cmp_sig2 = lp.pcoeff_burg(x, 10)
    assert_allclose(A, cmp_a, atol=5e-4)
    assert_allclose(SIG2, cmp_sig2, atol=1e-4)


@pytest.mark.parametrize("method, atol", [(lp.pcoeff_burg, 5e-3),
                                          (lp.pcoeff_covar, 1e-6)])
def test_simple_prediction(method, atol: float):
    """Extrapolate retarded Green's function of a box-like SIAM."""
    tt = np.linspace(0, 100, num=1001)
    # consider a box-like hybridization
    eps_0 = np.array(0.25)
    eps_b = np.linspace(-2, 2, num=1000)
    V = np.ones_like(eps_b)
    gf_ret_t = gt.siam.gf0_loc_ret_t(tt, eps_0, e_bath=eps_b, hopping=V)
    # try to extrapolate second half from first half
    gf_half = gf_ret_t[:tt.size//2+1]
    pcoeff, __ = method(gf_half, order=gf_half.size//2)
    gf_pred = lp.predict(gf_half, pcoeff=pcoeff, num=tt.size - gf_half.size)
    assert_allclose(gf_ret_t, gf_pred, atol=atol)


@pytest.mark.parametrize("fraction", [2, 3])
@pytest.mark.parametrize("lattice", [gt.lattice.bethe, gt.lattice.box])
@pytest.mark.parametrize("stable", [True, False])
def test_lattice_prediction(fraction, lattice, stable):
    """Test against continuous lattice Green's functions.

    Make sure that `stable=True` prediction is correct in the trivial case.
    """
    tt = np.linspace(0, 100, num=1001)
    gf_ret_t = lattice.gf_ret_t(tt, half_bandwidth=1, center=0.2)
    # try to extrapolate second half from first half
    gf_half = gf_ret_t[:tt.size//2+1]
    pcoeff, __ = lp.pcoeff_covar(gf_half, order=gf_half.size//fraction)
    gf_pred = lp.predict(gf_half, pcoeff=pcoeff, num=tt.size - gf_half.size,
                         stable=stable)
    assert_allclose(gf_ret_t, gf_pred, atol=3e-5)


@pytest.mark.parametrize("method, atol", [(lp.pcoeff_covar, 1e-6)])
@given(gufunc_args('(n),(n)->()', dtype=np.float_,
                   elements=[st.floats(min_value=-1, max_value=1),
                             st.floats(min_value=0, max_value=10),
                             ],
                   max_dims_extra=2, max_side=3, min_side=1),)
def test_predict_gufunc(method, atol, args):
    """Test that `predict` behaves like a proper gu-function.

    Currently uses `pcoeff_burg` as `pcoeff_covar` is not vectorized.
    """
    poles, resids = args
    assume(np.all(resids.sum(axis=-1) > 1e-4))
    # resids /= resids.sum(axis=-1, keepdims=True)
    # ensure sufficient large imaginary part
    tt = np.linspace(0, 20, 201)

    gf_t = gt.pole_gf_ret_t(tt, poles=poles[..., np.newaxis, :],
                            weights=resids[..., np.newaxis, :])

    gf_half = gf_t[..., :tt.size//2+1]
    pcoeff, __ = method(gf_half, order=gf_half.shape[-1]//2)
    gf_pred = lp.predict(gf_half, pcoeff=pcoeff, num=tt.size - gf_half.shape[-1])
    assert_allclose(gf_t, gf_pred, atol=atol)


def test_prediction_stability():
    """Test stability for box Green's functions.

    We add noise to generate growing exponentially growing terms.
    Prediction with `stable=True` should filter it and produce a somewhat
    reasonable prediction that doesn't explode.
    """
    lattice = gt.lattice.box
    tt = np.linspace(0, 100, num=1001)
    gf_ret_t = lattice.gf_ret_t(tt, half_bandwidth=1, center=0.2)
    # try to extrapolate second half from first half
    gf_half = gf_ret_t[:tt.size//2+1]
    noise = np.random.default_rng(0).normal(scale=1e-4, size=gf_half.size)
    pcoeff, __ = lp.pcoeff_covar(gf_half+noise, order=gf_half.size//2)
    # lp.plot_roots(pcoeff)
    gf_pred = lp.predict(gf_half, pcoeff=pcoeff, num=tt.size - gf_half.size)
    assert np.any(gf_pred > 1e3), "Make sure test is good, that is prediction grows"
    gf_pred = lp.predict(gf_half, pcoeff=pcoeff, num=tt.size - gf_half.size, stable=True)
    assert np.all(abs(gf_pred) <= 1), "Stable prediction should not grow"
    assert_allclose(gf_ret_t, gf_pred, atol=1e-2)
