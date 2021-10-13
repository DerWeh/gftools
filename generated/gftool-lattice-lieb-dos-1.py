eps = np.linspace(-1.5, 1.5, num=1001)
dos = gt.lattice.lieb.dos(eps, half_bandwidth=1)

import matplotlib.pyplot as plt
for pos in (-2**-0.5, 0, +2**-0.5):
    _ = plt.axvline(pos, color='black', linewidth=0.8)
_ = plt.plot(eps, dos)
_ = plt.xlabel(r"$\epsilon/D$")
_ = plt.ylabel(r"DOS * $D$")
_ = plt.ylim(bottom=0)
_ = plt.xlim(left=eps.min(), right=eps.max())
plt.show()
