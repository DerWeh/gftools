"""Basic implementation of Pade extracted from Junyas code as a reference.

notation: function u({iw}), a: coefficients for Pade, N: length of arrays

The algorithm is and exact implementation of the Vidberg & Serene paper.

"""
import numpy as np


# initialize `a`
def pade_init(iw, u, n_pade):
    """Return a_i = g_i(z_i).

    Parameters
    ----------
    iw :
        mesh on imaginary axis
    u :
        value of the function on iw-mesh
    n_pade :
        number of points used of pade

    Returns
    -------
    a :
        Pade coefficients

    """
    if iw.shape != u.shape:
        raise ValueError("Dimensions of `iw` and `u` don't match")
    if n_pade > len(iw):
        raise IndexError("`n_pade` larger then number of data points")
    g = np.zeros((n_pade, n_pade), dtype=np.complex256)
    iw = iw[:n_pade]
    g[0] = u[:n_pade]
    for i, gi in enumerate(g[1:]):
        gi[i+1:] = (g[i, i]/g[i, i+1:] - 1.) / (iw[i+1:] - iw[i])
    return g.diagonal()


def test_pade_init_junya(z, u, N):
    """Function copied from Junya's code as reference."""
    g = np.zeros((N, N), dtype=np.complex256)
    for i in range(N):
        g[0][i] = u[i]

    for n in range(1, N):
        for i in range(n, N):
            g[n][i] = (g[n-1, n-1]/g[n-1, i] - 1.) / (z[i] - z[n-1])

    a = np.zeros((N,), dtype=np.complex256)
    for i in range(N):
        a[i] = g[i, i]
    return a


# calculate u_pade(w) at w
def pade_calc(iw, a, w, n_pade):
    """Calculate Pade of function at points w.

    Parameters
    ----------
    iw :
        imaginary mesh used to calculate *a*
    a :
        coefficients for Pade, calculated from *pade_init*
    w :
        points at with the functions will be evaluated
    n_pade :
        number of imaginary frequencies used for the Pade

    Returns
    -------
    pade_calc :
        function evaluated at points *w*

    """
    A0, A1 = 0., a[0]
    B0, B1 = 1., 1.
    w = w.astype(dtype=np.complex256)
    for i in range(1, n_pade):
        A2 = A1 + (w - iw[i-1]) * a[i] * A0
        B2 = B1 + (w - iw[i-1]) * a[i] * B0
        A1 /= B2
        A2 /= B2
        B1 /= B2
        B2 /= B2

        A0 = A1
        A1 = A2
        B0 = B1
        B1 = B2
    return A2 / B2
