# Calculate integrals:

from mpmath import mp
mp.quad(gt.lattice.bethe.dos_mp, [-1, 1])
# mpf('1.0')

eps = np.linspace(-1.1, 1.1, num=500)
dos_mp = [gt.lattice.bethe.dos_mp(ee, half_bandwidth=1) for ee in eps]
dos_mp = np.array(dos_mp, dtype=np.float64)

import matplotlib.pyplot as plt
_ = plt.plot(eps, dos_mp)
_ = plt.xlabel(r"$\epsilon/D$")
_ = plt.ylabel(r"DOS * $D$")
_ = plt.axvline(0, color='black', linewidth=0.8)
_ = plt.ylim(bottom=0)
_ = plt.xlim(left=eps.min(), right=eps.max())
plt.show()
