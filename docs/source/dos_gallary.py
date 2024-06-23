"""Showcase of selection of DOSs."""
import matplotlib.pyplot as plt
import numpy as np

import gftool as gt


def atext(axis, text, x=0.97, y=0.85, fontsize="x-large", horizontalalignment="right", **kwds):
    """Add text legend on axis."""
    axis.text(x, y, s=text, transform=axis.transAxes, **kwds, fontsize=fontsize,
              horizontalalignment=horizontalalignment)


eps = np.linspace(-1.6, 1.6, num=5000)

_, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, dpi=200,
                       figsize=[12.8, 9.6])
for ax in axes.flat:
    ax.axvline(0, color="dimgray", linewidth=0.5)

axes[0, 0].plot(eps, gt.lattice.onedim.dos(eps, half_bandwidth=1))
atext(axes[0, 0], "1D")
axes[0, 1].plot(eps, gt.lattice.bethez.dos(eps, half_bandwidth=1, coordination=4))
atext(axes[0, 1], "Bethe $Z=4$")
axes[0, 2].plot(eps, gt.lattice.bethe.dos(eps, half_bandwidth=1))
atext(axes[0, 2], r"Bethe $Z=\infty$")

axes[1, 0].plot(eps, gt.lattice.square.dos(eps, half_bandwidth=1))
atext(axes[1, 0], "Square")
axes[1, 1].plot(eps, gt.lattice.triangular.dos(eps, half_bandwidth=1))
atext(axes[1, 1], "Triangular")
axes[1, 2].plot(eps, gt.lattice.honeycomb.dos(eps, half_bandwidth=1))
atext(axes[1, 2], "Honeycomb")

axes[2, 0].plot(eps, gt.lattice.sc.dos(eps, half_bandwidth=1))
atext(axes[2, 0], "Simple Cubic")
axes[2, 1].plot(eps, gt.lattice.bcc.dos(eps, half_bandwidth=1))
atext(axes[2, 1], "Body-Centered Cubic")
axes[2, 2].plot(eps, gt.lattice.fcc.dos(eps, half_bandwidth=1))
atext(axes[2, 2], "Face-Centered Cubic")

plt.xlim(eps[0], eps[-1])
plt.ylim(0, 1.75)
for ax in axes[-1]:
    ax.set_xlabel(r"$\epsilon/D$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"DOS$*D$")
plt.tight_layout(h_pad=0, w_pad=0)
plt.subplots_adjust(wspace=0, hspace=0, top=0.998, right=0.998)
plt.show()
