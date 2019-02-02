"""Test the functionality of the pade module."""
import pytest
import numpy as np

from . import old_pade

from .context import gftools as gt
from .context import gt_pade


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
    gf_bethe_iw = gt.bethe_gf_omega(iws, half_bandwidth=D) + pseudo_nois
    gf_bethe_iw_shift = gt.bethe_gf_omega(iws+shift, half_bandwidth=D) + pseudo_nois
    gf_bethe_fcts = np.array([gf_bethe_iw, gf_bethe_iw_shift])

    # compare sequential and parallel calculation
    coeff_single = np.array([gt_pade.coefficients(iws, fct_z=gf_bethe_iw),
                             gt_pade.coefficients(iws, fct_z=gf_bethe_iw_shift)])
    coeff_parall = gt_pade.coefficients(iws, fct_z=gf_bethe_fcts[np.newaxis])
    assert np.all(coeff_single == coeff_parall[0])
    avg_param = {'kind': gt_pade.KindGf(100, n_max), 'threshold': 1e-8}
    omega = np.linspace(-D*1.5, D*1.5, num=1000) + iws[0]/10
    avg_gf_z = [gt_pade.averaged(omega, z_in=iws, coeff=coeff_i, **avg_param)
                for coeff_i in coeff_single]
    avg_gf_z = gt.Result(x=np.array([avg_i.x for avg_i in avg_gf_z]),
                         err=np.array([avg_i.err for avg_i in avg_gf_z]))
    avg_gf_z_parall = gt_pade.averaged(omega, z_in=iws, coeff=coeff_parall, **avg_param)
    assert np.all(avg_gf_z_parall.x[0] == avg_gf_z.x)
    assert np.all(avg_gf_z_parall.err[0] == avg_gf_z.err)
