gf_coher_ww = g0(ww - self_cpa_ww)
__ = plt.plot(ww.real, -1/np.pi*gf_coher_ww.imag)
__ = plt.xlabel(r"$\omega$")
plt.show()
