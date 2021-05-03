eps = np.linspace(-1.5, 1.5, num=501)
dos = gt.lattice.honeycomb.dos(eps, half_bandwidth=1)

import matplotlib.pyplot as plt
for pos in (-1/3, 0, +1/3):
    _ = plt.axvline(pos, color='black', linewidth=0.8)
_ = plt.plot(eps, dos)
_ = plt.xlabel(r"$\epsilon/D$")
_ = plt.ylabel(r"DOS * $D$")
_ = plt.ylim(bottom=0)
_ = plt.xlim(left=eps.min(), right=eps.max())
plt.show()
