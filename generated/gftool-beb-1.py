from functools import partial

import gftool as gt
import numpy as np
import matplotlib.pyplot as plt

eps = np.array([-0.5, 0.5])
c = np.array([0.3, 0.7])
t = np.array([[1.0, 0.3],
              [0.3, 1.2]])
hilbert = partial(gt.bethe_hilbert_transform, half_bandwidth=1)

ww = np.linspace(-1.6, 1.6, num=1000) + 1e-4j
self_beb_ww = gt.beb.solve_root(ww, e_onsite=eps, concentration=c, hopping=t,
                                hilbert_trafo=hilbert)
gf_loc_ww = gt.beb.gf_loc_z(ww, self_beb_ww, hopping=t, hilbert_trafo=hilbert)

__ = plt.plot(ww.real, -1./np.pi/c[0]*gf_loc_ww[:, 0].imag, label='A')
__ = plt.plot(ww.real, -1./np.pi/c[1]*gf_loc_ww[:, 1].imag, label='B')
__ = plt.plot(ww.real, -1./np.pi*np.sum(gf_loc_ww.imag, axis=-1), ':', label='avg')
__ = plt.legend()
plt.show()