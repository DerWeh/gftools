# Calculated integrals

from mpmath import mp
mp.quad(gt.lattice.honeycomb.dos_mp, [-1, -1/3, 0, +1/3, +1])
# mpf('1.0')

eps = np.linspace(-1.5, 1.5, num=501)
dos_mp = [gt.lattice.honeycomb.dos_mp(ee, half_bandwidth=1) for ee in eps]

import matplotlib.pyplot as plt
for pos in (-1/3, 0, +1/3):
    _ = plt.axvline(pos, color='black', linewidth=0.8)
_ = plt.plot(eps, dos_mp)
_ = plt.xlabel(r"$\epsilon/D$")
_ = plt.ylabel(r"DOS * $D$")
_ = plt.ylim(bottom=0)
_ = plt.xlim(left=eps.min(), right=eps.max())
plt.show()
