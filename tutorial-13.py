from functools import partial
e_onsite = np.array([-0.3, -0.1, 0.4])
concentration = np.array([0.3, 0.5, 0.2])
hopping = np.ones([3, 3])
g0 = partial(gt.bethe_gf_z, half_bandwidth=1)
ww = np.linspace(-1.5, 1.5, num=501) + 1e-5j

self_cpa_ww = gt.cpa.solve_root(ww, e_onsite, concentration, hilbert_trafo=g0)
gf_coher_ww = g0(ww - self_cpa_ww)

self_beb_ww = gt.beb.solve_root(ww, e_onsite, concentration=concentration,
                                hopping=hopping, hilbert_trafo=g0)
gf_loc_ww = gt.beb.gf_loc_z(ww, self_beb_ww, hopping=hopping, hilbert_trafo=g0)

__ = plt.plot(ww.real, -1/np.pi*gf_coher_ww.imag, label="CPA avg")
__ = plt.plot(ww.real, -1/np.pi*gf_loc_ww.sum(axis=-1).imag,
             linestyle='--', label="BEB avg")
__ = plt.xlabel(r"$\omega$")
plt.show()
