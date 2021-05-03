eps = np.linspace(-1.5, 1.5, num=501)
bose = gt.bose_fct(eps, beta=1.0)

# The Bose function diverges at `eps=0`:

bose[eps==0]
# array([inf])

import matplotlib.pyplot as plt
_ = plt.plot(eps, bose)
_ = plt.xlabel(r"$\epsilon/\beta$")
_ = plt.axhline(0, color='black', linewidth=0.8)
_ = plt.axvline(0, color='black', linewidth=0.8)
_ = plt.xlim(left=eps.min(), right=eps.max())
plt.show()
