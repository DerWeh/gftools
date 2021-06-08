"""Test linear prediction."""
import numpy as np

from .context import gftool as gt

lp = gt.linearprediction


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
    assert np.allclose(A, cmp_a, atol=5e-4)
    assert np.allclose(SIG2, cmp_sig2, atol=1e-4)


def test_simple_extrapolation():
    """Extrapolate retarded Green's function of a box-like SIAM."""
    tt = np.linspace(0, 100, num=1001)
    # consider a box-like hybridization
    eps_0 = np.array(0.25)
    eps_b = np.linspace(-2, 2, num=1000)
    V = np.ones_like(eps_b)
    gf_ret_t = gt.siam.gf0_loc_ret_t(tt, eps_0, e_bath=eps_b, hopping=V)
    # try to extrapolate second half from first half
    gf_half = gf_ret_t[:tt.size//2+1]
    pcoeff, __ = lp.pcoeff_burg(gf_half, order=tt.size//4)
    gf_pred = lp.predict(gf_half, pcoeff=pcoeff, num=tt.size//2)
    np.allclose(gf_ret_t, gf_pred, atol=5e-3)
