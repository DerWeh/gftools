gf_cmpt_ww = gt.cpa.gf_cmpt_z(ww, self_cpa_ww, e_onsite, hilbert_trafo=g0)
c_gf_cmpt_ww = gf_cmpt_ww * concentration  # to compare with BEB
for cmpt in range(3):
    __ = plt.plot(ww.real, -1/np.pi*c_gf_cmpt_ww[..., cmpt].imag, label=f"CPA {cmpt}")
    __ = plt.plot(ww.real, -1/np.pi*gf_loc_ww[..., cmpt].imag, '--', label=f"BEB {cmpt}")
__ = plt.legend()
__ = plt.xlabel(r"$\omega$")
plt.show()
