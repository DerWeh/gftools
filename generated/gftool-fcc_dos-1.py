eps = np.linspace(-1.6, 1.6, num=501)
dos = gt.lattice.fcc.dos(eps, half_bandwidth=1)

import matplotlib.pyplot as plt
_ = plt.axvline(0, color='black', linewidth=0.8)
_ = plt.axvline(-0.5, color='black', linewidth=0.8)
_ = plt.plot(eps, dos)
_ = plt.xlabel(r"$\epsilon/D$")
_ = plt.ylabel(r"DOS * $D$")
_ = plt.ylim(bottom=0)
_ = plt.xlim(left=eps.min(), right=eps.max())
plt.show()
