ww = np.linspace(-1.5, 1.5, num=501) + 1e-6j
self_cpa_ww = gt.cpa.solve_root(ww, e_onsite, concentration, hilbert_trafo=g0)
