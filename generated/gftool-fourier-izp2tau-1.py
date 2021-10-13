BETA = 50
izp, __ = gt.pade_frequencies(50, beta=BETA)
tau = np.linspace(0, BETA, num=2049, endpoint=True)

poles = 2*np.random.random(10) - 1  # partially filled
weights = np.random.random(10)
weights = weights/np.sum(weights)
gf_izp = gt.pole_gf_z(izp, poles=poles, weights=weights)
gf_ft = gt.fourier.izp2tau(izp, gf_izp, tau, beta=BETA)
gf_tau = gt.pole_gf_tau(tau, poles=poles, weights=weights, beta=BETA)

import matplotlib.pyplot as plt
__ = plt.plot(tau, gf_tau, label='exact')
__ = plt.plot(tau, gf_ft, '--', label='FT')
__ = plt.legend()
plt.show()

__ = plt.title('Oscillations of tiny magnitude')
__ = plt.plot(tau/BETA, gf_tau - gf_ft)
__ = plt.xlabel('τ/β')
plt.show()

# Results of `izp2tau` can be improved giving high-frequency moments.

mom = np.sum(weights[:, np.newaxis] * poles[:, np.newaxis]**range(4), axis=0)
for n in range(1, 4):
    gf = gt.fourier.izp2tau(izp, gf_izp, tau, beta=BETA, moments=mom[:n])
    __ = plt.plot(tau, abs(gf_tau - gf), label=f'n_mom={n}')
__ = plt.legend()
plt.yscale('log')
plt.show()

# The method is resistant against noise:

magnitude = 2e-7
noise = np.random.normal(scale=magnitude, size=gf_izp.size)
gf = gt.fourier.izp2tau(izp, gf_izp + noise, tau, beta=BETA, moments=(1,))
__ = plt.plot(tau/BETA, abs(gf_tau - gf))
__ = plt.axhline(magnitude, color='black')
plt.yscale('log')
plt.tight_layout()
plt.show()

for n in range(1, 4):
    gf = gt.fourier.izp2tau(izp, gf_izp + noise, tau, beta=BETA, moments=mom[:n])
    __ = plt.plot(tau/BETA, abs(gf_tau - gf), '--', label=f'n_mom={n}')
__ = plt.axhline(magnitude, color='black')
__ = plt.plot(tau/BETA, abs(gf_tau - gf_ft), label='clean')
__ = plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.show()
