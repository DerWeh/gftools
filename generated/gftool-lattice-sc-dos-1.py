eps = np.linspace(-1.1, 1.1, num=501)
dos = gt.lattice.sc.dos(eps)

import matplotlib.pyplot as plt
_ = plt.axhline(0, color="black", linewidth=0.8)
_ = plt.axvline(-1/3, color="black", linewidth=0.8)
_ = plt.axvline(+1/3, color="black", linewidth=0.8)
_ = plt.plot(eps, dos)
_ = plt.xlabel(r"$\epsilon/D$")
_ = plt.ylabel(r"DOS * $D$")
_ = plt.xlim(left=eps.min(), right=eps.max())
plt.show()
