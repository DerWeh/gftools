# Calculate integrals (the 1D DOS needs higher accuracy for accurate results):

from mpmath import mp
with mp.workdps(35, normalize_output=True):
    norm = mp.quad(gt.lattice.onedim.dos_mp, [-1, +1])
norm
# mpf('1.0')

eps = np.linspace(-1.1, 1.1, num=501)
dos_mp = [gt.lattice.onedim.dos_mp(ee, half_bandwidth=1) for ee in eps]

import matplotlib.pyplot as plt
_ = plt.plot(eps, dos_mp)
for pos in (-1, 0, +1):
    _ = plt.axvline(pos, color='black', linewidth=0.8)
_ = plt.xlabel(r"$\epsilon/D$")
_ = plt.ylabel(r"DOS * $D$")
_ = plt.ylim(bottom=0)
_ = plt.xlim(left=eps.min(), right=eps.max())
plt.show()
