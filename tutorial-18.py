__ = plt.plot(ww.real, -1.0/np.pi*gf_ww.imag[:, N//2, N//2, N//2, N//2])
__ = plt.plot(ww.real, -1.0/np.pi*gt.square_gf_z(ww, half_bandwidth=4*t).imag,
              color='black', linestyle='--')
__ = plt.xlabel(r"$\omega$")
plt.show()
