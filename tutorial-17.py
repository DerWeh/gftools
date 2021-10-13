dec = gt.matrix.decompose_her(ham_mat)
ww = np.linspace(-2.5, 2.5, num=201) + 1e-1j  # frequency match
gf_ww = dec.reconstruct(1.0/(ww[:, np.newaxis] - dec.eig))
gf_ww = gf_ww.reshape(ww.size, *[N]*4)  # reshape for easy access
