from functools import partial
parameter = dict(
    e_onsite=[-0.3, 0.3],
    concentration=[0.3, 0.7],
    hilbert_trafo=partial(gt.bethe_gf_z, half_bandwidth=1),
)

ww = np.linspace(-1.5, 1.5, num=5000) + 1e-10j
self_cpa_ww = gt.cpa.solve_root(ww, **parameter)
del parameter['concentration']
gf_cmpt_ww = gt.cpa.gf_cmpt_z(ww, self_cpa_ww, **parameter)

import matplotlib.pyplot as plt
__ = plt.plot(ww.real, -1./np.pi*gf_cmpt_ww[..., 0].imag)
__ = plt.plot(ww.real, -1./np.pi*gf_cmpt_ww[..., 1].imag)
plt.show()
