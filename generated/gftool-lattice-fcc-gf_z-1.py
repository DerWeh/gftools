ww = np.linspace(-1.6, 1.6, num=501, dtype=complex)
gf_ww = gt.lattice.fcc.gf_z(ww, half_bandwidth=1)

import matplotlib.pyplot as plt
_ = plt.axvline(-0.5, color='black', linewidth=0.8)
_ = plt.axvline(0, color='black', linewidth=0.8)
_ = plt.axhline(0, color='black', linewidth=0.8)
_ = plt.plot(ww.real, gf_ww.real, label=r"$\Re G$")
_ = plt.plot(ww.real, gf_ww.imag, '--', label=r"$\Im G$")
_ = plt.ylabel(r"$G*D$")
_ = plt.xlabel(r"$\omega/D$")
_ = plt.xlim(left=ww.real.min(), right=ww.real.max())
_ = plt.legend()
plt.show()
