gf_iw = gt.bethe_gf_z(iws + eps[:, np.newaxis], half_bandwidth=1)
gf_iw.shape
# (2, 1024)
gf_tau = gt.fourier.iw2tau(gf_iw, beta=beta)
