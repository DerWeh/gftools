# TODO
# from pathlib import Path
# from sys import path as syspath
# syspath.insert(1, str(Path('.').absolute().parent))
# import gftools as gt
# dir_ = Path('~/workspace/articles/master_thesis/data/m_hfm_m/multilayer/t02_mu045_h-9_UX/U08/1hf_19m_U08_poisson/output/00-G_omega.dat').expanduser()
# test_data = np.loadtxt(dir_, unpack=True)
# beta = 1. / 0.01
# iws = gt.matsubara_frequencies(test_data[0], beta=beta)
# omega = np.linspace(-4, 4., num=1000) + 1e-6j
# gf_mid_iw = test_data[1] + 1j*test_data[2]
# coeff = coefficients(iws, gf_mid_iw)
# pade = np.array([calc(omega, iws, coeff, n_max=n) for n in range(590, 600, 2)])
# pade2 = np.array([pade_ for pade_ in calc_iterator(omega, iws, coeff, n_min=589, n_max=599)])
