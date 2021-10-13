eps = np.linspace(-15, 15, num=501)
fermi_d1 = gt.fermi_fct_d1(eps, beta=1.0)

import matplotlib.pyplot as plt
_ = plt.plot(eps, fermi_d1)
_ = plt.xlabel(r"$\epsilon/\beta$")
_ = plt.axvline(0, color='black', linewidth=0.8)
_ = plt.xlim(left=eps.min(), right=eps.max())
_ = plt.ylim(top=0)
plt.show()
