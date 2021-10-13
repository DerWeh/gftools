e_onsite = np.array([[-0.3, +0.15, +0.4],
                     [-0.3, -0.35, +0.4]])
concentration = np.array([0.3, 0.5, 0.2])
g0 = partial(gt.fcc_gf_z, half_bandwidth=1)
self_cpa_ww = gt.cpa.solve_root(ww[:, np.newaxis], e_onsite, concentration,
                                hilbert_trafo=g0, options=dict(fatol=1e-8),
                                self_cpa_z0=self_cpa_ww[:, np.newaxis])
gf_cmpt_ww = gt.cpa.gf_cmpt_z(ww[:, np.newaxis], self_cpa_ww, e_onsite, hilbert_trafo=g0)
__, axes = plt.subplots(nrows=2, sharex=True)
for spin, ax in enumerate(axes):
    for cmpt in range(3):
        __ = ax.plot(ww.real, -1/np.pi*gf_cmpt_ww[:, spin, cmpt].imag, label=f"cmpt {cmpt}")
__ = plt.legend()
__ = plt.xlabel(r"$\omega$")
plt.show()
