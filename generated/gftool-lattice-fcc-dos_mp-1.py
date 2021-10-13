# Calculate integrals:

from mpmath import mp
unit = mp.quad(gt.lattice.fcc.dos_mp, [-0.5, 0, 1.5])
mp.identify(unit)
# '1'

eps = np.linspace(-1.6, 1.6, num=501)
dos_mp = [gt.lattice.fcc.dos_mp(ee, half_bandwidth=1) for ee in eps]
dos_mp = np.array(dos_mp, dtype=np.float64)

import matplotlib.pyplot as plt
_ = plt.axvline(0, color='black', linewidth=0.8)
_ = plt.axvline(-0.5, color='black', linewidth=0.8)
_ = plt.plot(eps, dos_mp)
_ = plt.xlabel(r"$\epsilon/D$")
_ = plt.ylabel(r"DOS * $D$")
_ = plt.ylim(bottom=0)
_ = plt.xlim(left=eps.min(), right=eps.max())
plt.show()
