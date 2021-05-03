# Calculated integrals

from mpmath import mp
mp.identify(mp.quad(gt.lattice.kagome.dos_mp, [-2/3, 0, 1/3, 2/3, 4/3]))
# '(2/3)'

eps = np.linspace(-1.5, 1.5, num=1001)
dos_mp = [gt.lattice.kagome.dos(ee, half_bandwidth=1) for ee in eps]

import matplotlib.pyplot as plt
for pos in (-2/3, 0, +2/3):
    _ = plt.axvline(pos, color='black', linewidth=0.8)
_ = plt.plot(eps, dos_mp)
_ = plt.xlabel(r"$\epsilon/D$")
_ = plt.ylabel(r"DOS * $D$")
_ = plt.ylim(bottom=0)
_ = plt.xlim(left=eps.min(), right=eps.max())
plt.show()
