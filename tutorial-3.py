beta = 50
iws = gt.matsubara_frequencies(range(128), beta=beta)
gf_iw = gt.lattice.fcc.gf_z(iws, half_bandwidth=1.)
__ = plt.axhline(0, color="dimgray", linewidth=0.8)
__ = plt.plot(gf_iw.real, "x--")
__ = plt.plot(gf_iw.imag, "+--")
__ = plt.xlabel("$n$")
plt.show()
