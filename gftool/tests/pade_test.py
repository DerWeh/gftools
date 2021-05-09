"""Test the functionality of the pade module."""
import logging

import numpy as np

from . import old_pade
from .context import gftool as gt, gt_pade

logging.basicConfig(level=logging.DEBUG)


def test_regression():
    """Compare the improved methods to the standard implementation."""
    # check coefficients
    T = 0.037
    D = 1.2
    iws = gt.matsubara_frequencies(np.arange(2**10), beta=1./T)
    # use quad precision for comparability
    iws = iws.astype(dtype=np.complex256)
    rand = np.random.random(iws.size) + 1j*np.random.random(iws.size)
    rand *= 1e-3
    rand *= 1 + np.sqrt(np.arange(iws.size))
    gf_bethe_iw = gt.bethe_gf_z(iws, half_bandwidth=D) + rand
    coeff_old = old_pade.test_pade_init_junya(iws, u=gf_bethe_iw, N=iws.size)
    coeff = gt_pade.coefficients(iws, fct_z=gf_bethe_iw)
    assert np.all(coeff_old == coeff)
    omega = np.linspace(-D*1.2, D*1.2, num=1000) + iws[0]/10
    kind = gt_pade.KindGf(iws.size-4, iws.size-2)
    # FIXME: look into different counting
    gf_bethe_old = old_pade.pade_calc(iw=iws, a=coeff_old, w=omega, n_pade=kind[-1]+2)
    gf_bethe = list(kind.islice(gt_pade.calc_iterator(omega, z_in=iws, coeff=coeff)))[-1]
    assert np.allclose(gf_bethe_old, gf_bethe, rtol=1e-14, atol=1e-14)


def test_coeff_type_reduction():
    """Check that `coefficients` returns complex128 if input is only double precision."""
    T = 0.037
    D = 1.2
    iws = gt.matsubara_frequencies(np.arange(10), beta=1./T)
    gf_bethe_iw = gt.bethe_gf_z(iws, half_bandwidth=D)
    coeff = gt_pade.coefficients(iws, fct_z=gf_bethe_iw.astype(dtype=np.complex128))
    assert coeff.dtype == np.dtype(np.complex128)
    coeff = gt_pade.coefficients(iws, fct_z=gf_bethe_iw.astype(dtype=np.complex256))
    assert coeff.dtype == np.dtype(np.complex256)


def passing():  # compare calculation (masked) to exact result
    import matplotlib.pyplot as plt

    plt.figure('diff')
    plt.plot(coeff_old.imag - coeff.imag)
    plt.figure('coefficients')
    plt.plot(coeff_old.imag, label='old')
    plt.plot(coeff.imag, '--', label='new')
    plt.legend()
    # plt.plot(coeff_old.real)
    # plt.plot(coeff.real, '--')

    # show results
    # omega = np.linspace(-D*2, D*2, num=1000) + 1e-6j
    omega = np.linspace(-D*1.2, D*1.2, num=1000) + iws[0]/10
    gf_bethe_w1 = old_pade.pade_calc(iw=iws, a=coeff_old, w=omega, n_pade=iws.size)
    gf_bethe_w2 = old_pade.pade_calc(iw=iws, a=coeff, w=omega, n_pade=iws.size)
    plt.figure('compare')
    plt.plot(omega.real, -gf_bethe_w1.imag, label='odl')
    plt.plot(omega.real, -gf_bethe_w2.imag, '--', label='new')
    plt.plot(omega.real, -gt.bethe_gf_omega(omega, half_bandwidth=D).imag, '-.', label='exact')
    plt.legend()
    plt.show()


def test_stacked_pade():
    """Test results of calculating stacked Pades in parallel against single Pade."""
    # Test setup
    T = 0.037
    D = 1.2
    shift = 0.48
    n_max = 800
    iws = gt.matsubara_frequencies(np.arange(n_max), beta=1./T)
    # disturb results to have invalid values in averaging
    pseudo_nois = np.sin(iws.imag) * 2e-4
    gf_bethe_iw = gt.bethe_gf_z(iws, half_bandwidth=D) + pseudo_nois
    gf_bethe_iw_shift = gt.bethe_gf_z(iws+shift, half_bandwidth=D) + pseudo_nois
    gf_bethe_fcts = np.array([gf_bethe_iw, gf_bethe_iw_shift])

    # compare sequential and parallel calculation
    coeff_single = np.array([gt_pade.coefficients(iws, fct_z=gf_bethe_iw),
                             gt_pade.coefficients(iws, fct_z=gf_bethe_iw_shift)])
    coeff_parall = gt_pade.coefficients(iws, fct_z=gf_bethe_fcts[np.newaxis])
    assert np.all(coeff_single == coeff_parall[0])
    avg_param = {'kind': gt_pade.KindGf(100, n_max),
                 'filter_valid': gt_pade.FilterNegImag(1e-8)}
    omega = np.linspace(-D*1.5, D*1.5, num=1000) + iws[0]/10
    avg_gf_z = [gt_pade.averaged(omega, z_in=iws, coeff=coeff_i, **avg_param)
                for coeff_i in coeff_single]
    avg_gf_z = gt.Result(x=np.array([avg_i.x for avg_i in avg_gf_z]),
                         err=np.array([avg_i.err for avg_i in avg_gf_z]))
    avg_gf_z_parall = gt_pade.averaged(omega, z_in=iws, coeff=coeff_parall, **avg_param)
    assert np.all(avg_gf_z_parall.x[0] == avg_gf_z.x)
    assert np.all(avg_gf_z_parall.err[0] == avg_gf_z.err)
