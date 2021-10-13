U = 5
mu = U/2  # particle-hole symmetric case -> n=0.5
ww = np.linspace(-5, 5, num=1000) + 1e-6j

self_ww = gt.hubbard_I_self_z(ww+mu, U, occ=0.5)

# Show the spectral function for the Bethe lattice,
# we see the two Hubbard bands centered at ±U/2:

import matplotlib.pyplot as plt
gf_iw = gt.bethe_gf_z(ww+mu-self_ww, half_bandwidth=1.)
__ = plt.plot(ww.real, -1./np.pi*gf_iw.imag)
plt.show()
