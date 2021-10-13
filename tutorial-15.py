from functools import partial
e_onsite = np.array([0.2, -0.2])
concentration = np.array([0.3, 0.7])
hopping = np.array([[1.0, 0.5],
                    [0.5, 1.2]])
g0 = partial(gt.bethe_gf_z, half_bandwidth=1)
ww = np.linspace(-1.5, 1.5, num=501) + 1e-5j

self_beb_ww = gt.beb.solve_root(ww, e_onsite, concentration=concentration,
                                hopping=hopping, hilbert_trafo=g0)
gf_loc_ww = gt.beb.gf_loc_z(ww, self_beb_ww, hopping=hopping, hilbert_trafo=g0)
__ = plt.plot(ww.real, -1/np.pi*gf_loc_ww[..., 0].imag, label="A")
__ = plt.plot(ww.real, -1/np.pi*gf_loc_ww[..., 1].imag, label="B")
__ = plt.plot(ww.real, -1/np.pi*gf_loc_ww.sum(axis=-1).imag,
             linestyle='--', label="BEB avg")
__ = plt.legend()
__ = plt.xlabel(r"$\omega$")
plt.show()
