# Calculated integrals

from mpmath import mp
mp.identify(mp.quad(gt.lattice.lieb.dos_mp, [-1, -2**-0.5, 0, 2**-0.5, 1]))
# '(2/3)'

eps = np.linspace(-1.5, 1.5, num=501)
dos_mp = [gt.lattice.lieb.dos_mp(ee, half_bandwidth=1) for ee in eps]

import matplotlib.pyplot as plt
for pos in (-2**-0.5, 0, +2**-0.5):
    _ = plt.axvline(pos, color='black', linewidth=0.8)
_ = plt.plot(eps, dos_mp)
_ = plt.xlabel(r"$\epsilon/D$")
_ = plt.ylabel(r"DOS * $D$")
_ = plt.ylim(bottom=0)
_ = plt.xlim(left=eps.min(), right=eps.max())
plt.show()
