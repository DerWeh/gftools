ww = np.linspace(-1.5, 1.5, num=500)
gf_ww = gt.lattice.onedim.gf_z(ww, half_bandwidth=1)

import matplotlib.pyplot as plt
_ = plt.axhline(0, color='black', linewidth=0.8)
_ = plt.plot(ww, gf_ww.real, label=r"$\Re G$")
_ = plt.plot(ww, gf_ww.imag, '--', label=r"$\Im G$")
_ = plt.xlabel(r"$\omega/D$")
_ = plt.ylabel("G*D")
_ = plt.xlim(left=ww.min(), right=ww.max())
_ = plt.legend()
plt.show()
