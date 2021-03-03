r"""2D triangular lattice.

The dispersion of the 2D triangular lattice is given by

.. math:: ϵ_{k_x, k_y} = t [\cos(2k_x) + 2 \cos(k_x)\cos(k_y)]

:half_bandwidth: The half-bandwidth `D` corresponds to a nearest neighbor hopping
                 of FIXME:`t=D/3`.

"""
import numpy as np

from gftool._util import _u_ellipk


def gf_z(z, half_bandwidth):
    del half_bandwidth
    sqrt = (2.0*z + 3)**0.5
    g = 4.0 / ((sqrt - 1)**1.5 * (sqrt + 3.0)**0.5)
    ksqr = sqrt * g**2
    elliptic = _u_ellipk(ksqr)
    gf_z = 1 / np.pi * g * elliptic
    return gf_z
