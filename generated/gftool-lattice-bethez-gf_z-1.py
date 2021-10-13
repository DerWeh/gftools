ww = np.linspace(-1.5, 1.5, num=500)
gf_ww = gt.lattice.bethez.gf_z(ww, half_bandwidth=1, coordination=9)

import matplotlib.pyplot as plt
_ = plt.plot(ww, gf_ww.real, label=r"$\Re G$")
_ = plt.plot(ww, gf_ww.imag, '--', label=r"$\Im G$")
_ = plt.xlabel(r"$\omega/D$")
_ = plt.ylabel(r"$G*D$")
_ = plt.axhline(0, color='black', linewidth=0.8)
_ = plt.xlim(left=ww.min(), right=ww.max())
_ = plt.legend()
plt.show()
