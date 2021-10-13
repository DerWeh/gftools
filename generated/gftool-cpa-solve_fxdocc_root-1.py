from functools import partial
beta = 30
e_onsite = [-0.3, 0.3]
conc = [0.3, 0.7]
hilbert = partial(gt.bethe_gf_z, half_bandwidth=1)
occ = 0.5,

iws = gt.matsubara_frequencies(range(1024), beta=30)
self_cpa_iw, mu = gt.cpa.solve_fxdocc_root(iws, e_onsite, conc,
                                           hilbert, occ=occ, beta=beta)

import matplotlib.pyplot as plt
__ = plt.plot(iws.imag, self_cpa_iw.imag, '+--')
__ = plt.axhline(np.average(e_onsite, weights=conc) - mu)
__ = plt.plot(iws.imag, self_cpa_iw.real, 'x--')
plt.show()

# check occupation

gf_coher_iw = hilbert(iws - self_cpa_iw)
gt.density_iw(iws, gf_coher_iw, beta=beta, moments=[1, self_cpa_iw[-1].real])
# 0.499999...

# check CPA

self_compare = gt.cpa.solve_root(iws, np.array(e_onsite)-mu, conc,
                                 hilbert_trafo=hilbert)
np.allclose(self_cpa_iw, self_compare, atol=1e-5)
# True
