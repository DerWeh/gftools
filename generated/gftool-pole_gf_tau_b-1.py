beta = 10
tau = np.linspace(0, beta, num=1000)
gf_tau = gt.pole_gf_tau_b(tau, .5, 1., beta=beta)

# The integrated imaginary time Green's function gives `-np.sum(weights/poles)`

np.trapz(gf_tau, x=tau)
# -2.0000041750107735

import matplotlib.pyplot as plt
__ = plt.plot(tau, gf_tau)
__ = plt.xlabel('τ')
plt.show()
