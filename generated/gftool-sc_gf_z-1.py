ww = np.linspace(-1.1, 1.1, num=500)
gf_ww = gt.lattice.sc.gf_z(ww)

import matplotlib.pyplot as plt
_ = plt.axhline(0, color="black", linewidth=0.8)
_ = plt.axvline(-1/3, color="black", linewidth=0.8)
_ = plt.axvline(+1/3, color="black", linewidth=0.8)
_ = plt.plot(ww.real, gf_ww.real, label=r"$\Re G$")
_ = plt.plot(ww.real, gf_ww.imag, label=r"$\Im G$")
_ = plt.ylabel(r"$G*D$")
_ = plt.xlabel(r"$\omega/D$")
_ = plt.xlim(left=ww.min(), right=ww.max())
_ = plt.legend()
plt.show()
