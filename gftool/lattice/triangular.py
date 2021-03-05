r"""2D triangular lattice.

The dispersion of the 2D triangular lattice is given by

.. math:: ϵ_{k_x, k_y} = t [\cos(2k_x) + 2 \cos(k_x)\cos(k_y)]

:half_bandwidth: The half-bandwidth `D` corresponds to a nearest neighbor hopping
                 of FIXME:`t=D/3`.

"""
import numpy as np

from gftool._util import _u_ellipk


def _signed_sqrt(z):
    """Square root with correct sign for triangular lattice."""
    sign = np.where((z.real < 0) & (z.imag < 0), -1, 1)
    return sign * np.lib.scimath.sqrt(z)


def gf_z(z, half_bandwidth):
    del half_bandwidth
    z = np.asarray(z)
    shape = z.shape
    z = z.reshape(-1)
    advanced = z.imag < 0
    z = np.where(advanced, np.conj(z), z)  # calculate retarded only, and use symmetry
    singular = z == -1  # try singularity which needs to be avoided
    z[singular] = 0  # mock value to avoid errors
    rr = _signed_sqrt(2*z + 3)
    gg = 4.0 / (_signed_sqrt(rr - 1)**3 * _signed_sqrt(rr + 3))  # eq (2.9)
    kk = _signed_sqrt(rr) * gg  # eq (2.11)
    mm = kk**2
    K = np.asarray(_u_ellipk(mm))
    # eqs (2.22) and eq (2.18), fix correct plane
    K[kk.imag > 0] += 2j*_u_ellipk(1 - mm[kk.imag > 0])
    gf_z = 1 / np.pi * gg * K  # eq (2.6)
    gf_z[singular] = 0 - 1j*np.infty
    return np.where(advanced, np.conj(gf_z), gf_z).reshape(shape)  # return to advanced by symmetry
