BETA = 50
tau = np.linspace(0, BETA, num=2049, endpoint=True)
ivs = gt.matsubara_frequencies_b(range((tau.size+1)//2), beta=BETA)

poles, weights = np.random.random(10), np.random.random(10)
weights = weights/np.sum(weights)
gf_tau = gt.pole_gf_tau_b(tau, poles=poles, weights=weights, beta=BETA)
gf_ft = gt.fourier.tau2iv(gf_tau, beta=BETA)
gf_tau.size, gf_ft.size
# (2049, 1025)
gf_iv = gt.pole_gf_z(ivs, poles=poles, weights=weights)

import matplotlib.pyplot as plt
__ = plt.plot(gf_iv.imag, label='exact Im')
__ = plt.plot(gf_ft.imag, '--', label='DFT Im')
__ = plt.plot(gf_iv.real, label='exact Re')
__ = plt.plot(gf_ft.real, '--', label='DFT Re')
__ = plt.legend()
plt.show()

# Accuracy of the different back-ends

ft_lin, dft = gt.fourier.tau2iv_ft_lin, gt.fourier.tau2iv_dft
gf_ft_lin = gt.fourier.tau2iv(gf_tau, beta=BETA, fourier=ft_lin)
gf_dft = gt.fourier.tau2iv(gf_tau, beta=BETA, fourier=dft)
__ = plt.plot(abs(gf_iv - gf_ft_lin), label='FT_lin')
__ = plt.plot(abs(gf_iv - gf_dft), '--', label='DFT')
__ = plt.legend()
plt.yscale('log')
plt.show()

# The methods are resistant against noise:

magnitude = 5e-6
noise = np.random.normal(scale=magnitude, size=gf_tau.size)
gf_ft_lin_noisy = gt.fourier.tau2iv(gf_tau + noise, beta=BETA, fourier=ft_lin)
gf_dft_noisy = gt.fourier.tau2iv(gf_tau + noise, beta=BETA, fourier=dft)
__ = plt.plot(abs(gf_iv - gf_ft_lin_noisy), '--', label='FT_lin')
__ = plt.plot(abs(gf_iv - gf_dft_noisy), '--', label='DFT')
__ = plt.axhline(magnitude, color='black')
__ = plt.legend()
plt.yscale('log')
plt.show()
