self_cpa_ww = gt.cpa.solve_root(ww, e_onsite, concentration, hilbert_trafo=g0,
                                options=dict(fatol=1e-10))
gf_coher_ww = g0(ww - self_cpa_ww)
__ = plt.plot(ww.real, -1/np.pi*gf_coher_ww.imag)
__ = plt.xlabel(r"$\omega$")
plt.show()
