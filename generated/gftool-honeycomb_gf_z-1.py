ww = np.linspace(-1.5, 1.5, num=501, dtype=complex) + 1e-64j
gf_ww = gt.lattice.honeycomb.gf_z(ww, half_bandwidth=1)

import matplotlib.pyplot as plt
_ = plt.axhline(0, color='black', linewidth=0.8)
_ = plt.plot(ww.real, gf_ww.real, label=r"$\Re G$")
_ = plt.plot(ww.real, gf_ww.imag, '--', label=r"$\Im G$")
_ = plt.ylabel(r"$G*D$")
_ = plt.xlabel(r"$\omega/D$")
_ = plt.xlim(left=ww.real.min(), right=ww.real.max())
_ = plt.legend()
plt.show()
