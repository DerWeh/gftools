BETA = 50
tau = np.linspace(0, BETA, num=2049, endpoint=True)
ivs = gt.matsubara_frequencies_b(range((tau.size+1)//2), beta=BETA)

poles, weights = np.random.random(10), np.random.random(10)
weights = weights/np.sum(weights)
gf_tau = gt.pole_gf_tau(tau, poles=poles, weights=weights, beta=BETA)
gf_dft = gt.fourier.tau2iv_dft(gf_tau, beta=BETA)
gf_tau.size, gf_dft.size
# (2049, 1025)
gf_iv = gt.pole_gf_z(ivs, poles=poles, weights=weights)

import matplotlib.pyplot as plt
__ = plt.plot(gf_iv.imag, label='exact Im')
__ = plt.plot(gf_dft.imag, '--', label='DFT Im')
__ = plt.plot(gf_iv.real, label='exact Re')
__ = plt.plot(gf_dft.real, '--', label='DFT Re')
__ = plt.legend()
plt.show()

__ = plt.title('Error growing with frequency')
__ = plt.plot(abs(gf_iv - gf_dft))
plt.yscale('log')
plt.show()

# The method is resistant against noise:

magnitude = 2e-3
noise = np.random.normal(scale=magnitude, size=gf_tau.size)
gf_dft_noisy = gt.fourier.tau2iv_dft(gf_tau + noise, beta=BETA)
__ = plt.plot(abs(gf_iv - gf_dft_noisy), '--', label='noisy')
__ = plt.axhline(magnitude, color='black')
__ = plt.plot(abs(gf_iv - gf_dft), label='clean')
__ = plt.legend()
# plt.yscale('log')
plt.show()
