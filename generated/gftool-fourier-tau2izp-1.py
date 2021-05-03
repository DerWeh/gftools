BETA = 50
tau = np.linspace(0, BETA, num=2049, endpoint=True)
izp, __ = gt.pade_frequencies(50, beta=BETA)

poles = 2*np.random.random(10) - 1  # partially filled
weights = np.random.random(10)
weights = weights/np.sum(weights)
gf_tau = gt.pole_gf_tau(tau, poles=poles, weights=weights, beta=BETA)
gf_ft = gt.fourier.tau2izp(gf_tau, beta=BETA, izp=izp)
gf_izp = gt.pole_gf_z(izp, poles=poles, weights=weights)

import matplotlib.pyplot as plt
__ = plt.plot(gf_izp.imag, label='exact Im')
__ = plt.plot(gf_ft.imag, '--', label='FT Im')
__ = plt.plot(gf_izp.real, label='exact Re')
__ = plt.plot(gf_ft.real, '--', label='FT Re')
__ = plt.legend()
plt.show()

# Results of `tau2izp` can be improved giving high-frequency moments.

mom = np.sum(weights[:, np.newaxis] * poles[:, np.newaxis]**range(6), axis=0)
for n in range(1, 6):
    gf = gt.fourier.tau2izp(gf_tau, izp=izp, moments=mom[:n], beta=BETA)
    __ = plt.plot(abs(gf_izp - gf), label=f'n_mom={n}', color=f'C{n}')
__ = plt.legend()
plt.yscale('log')
plt.show()

# The method is resistant against noise,
# especially if there is knowledge of the noise:

magnitude = 2e-7
noise = np.random.normal(scale=magnitude, size=gf_tau.size)
gf = gt.fourier.tau2izp(gf_tau + noise, izp=izp, moments=(1,), beta=BETA)
__ = plt.plot(abs(gf_izp - gf), label='bare')
gf = gt.fourier.tau2izp(gf_tau + noise, izp=izp, moments=(1,), beta=BETA,
                        weight=abs(noise)**-0.5)
__ = plt.plot(abs(gf_izp - gf), label='weighted')
__ = plt.axhline(magnitude, color='black')
__ = plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.show()

for n in range(1, 7, 2):
    gf = gt.fourier.tau2izp(gf_tau + noise, izp=izp, moments=mom[:n], beta=BETA)
    __ = plt.plot(abs(gf_izp - gf), '--', label=f'n_mom={n}', color=f'C{n}')
__ = plt.axhline(magnitude, color='black')
__ = plt.plot(abs(gf_izp - gf_ft), label='clean')
__ = plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.show()
