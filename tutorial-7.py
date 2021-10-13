from functools import partial
e_onsite = np.array([-0.3, -0.1, 0.4])
concentration = np.array([0.3, 0.5, 0.2])
g0 = partial(gt.bethe_gf_z, half_bandwidth=1)
