ww = np.linspace(-0.9, 1.7, num=1000) + 1e-6j
gf_ww = gt.lattice.fcc.gf_z(ww, half_bandwidth=1.)
__ = plt.axhline(0, color="dimgray", linewidth=0.8)
__ = plt.axvline(0, color="dimgray", linewidth=0.8)
__ = plt.plot(ww.real, gf_ww.real)
__ = plt.plot(ww.real, gf_ww.imag)
__ = plt.xlabel(r"$\omega$")
__ = plt.ylim(-7.0, 2.5)
plt.show()
