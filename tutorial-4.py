beta = 20  # inverse temperature
h = 0.3  # magnetic splitting
eps = np.array([-0.5*h, +0.5*h])  # on-site energy
iws = gt.matsubara_frequencies(range(1024), beta=beta)
