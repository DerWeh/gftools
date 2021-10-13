N = 21   # system size in one dimension
t = tx = ty = 0.5  # hopping amplitude
hamilton = np.zeros([N]*4)
diag = np.arange(N)
hamilton[diag[1:], :, diag[:-1], :] = hamilton[diag[:-1], :, diag[1:], :] = -tx
hamilton[:, diag[1:], :, diag[:-1]] = hamilton[:, diag[:-1], :, diag[1:]] = -ty
ham_mat = hamilton.reshape(N**2, N**2)  # turn in into a matrix
