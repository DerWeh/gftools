gf_cmpt_ww = gt.cpa.gf_cmpt_z(ww, self_cpa_ww, e_onsite, hilbert_trafo=g0)
np.allclose(np.sum(concentration*gf_cmpt_ww, axis=-1), gf_coher_ww)
# True
for cmpt in range(3):
    __ = plt.plot(ww.real, -1/np.pi*gf_cmpt_ww[..., cmpt].imag, label=f"cmpt {cmpt}")
__ = plt.plot(ww.real, -1/np.pi*gf_coher_ww.imag, linestyle=':', label="avg")
__ = plt.legend()
__ = plt.xlabel(r"$\omega$")
plt.show()
