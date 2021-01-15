"""Pre-compute the simple cubic DOS once and for all."""
from functools import partial
import numpy as np

from mpmath import mp
from tqdm import tqdm

from gftool.lattice import scubic
from gftool.basis.pole import _chebyshev_points

dos = partial(scubic.dos_mp, half_bandwidth=1, maxdegree=12)

num = 500
vanhow = 1.0/3.0

dom1_che = (_chebyshev_points(num=num) + 1)/6
dom1_lin = np.linspace(0, vanhow, num=num)
dom2_che = (_chebyshev_points(num=2*num) + 2)/3
dom2_lin = np.linspace(vanhow, 1, num=2*num)

with mp.workdps(50):
    dos_dom1_che = np.array([dos(eps) for eps in tqdm(dom1_che)], dtype=np.float_)
    np.savez("dom1_che", x=dom1_che, dos=dos_dom1_che[:, 0], err=dos_dom1_che[:, 1])
    dos_dom1_lin = np.array([dos(eps) for eps in tqdm(dom1_lin)], dtype=np.float_)
    np.savez("dom1_lin", x=dom1_lin, dos=dos_dom1_lin[:, 0], err=dos_dom1_lin[:, 1])
    dos_dom2_che = np.array([dos(eps) for eps in tqdm(dom2_che)], dtype=np.float_)
    np.savez("dom2_che", x=dom2_che, dos=dos_dom2_che[:, 0], err=dos_dom2_che[:, 1])
    dos_dom2_lin = np.array([dos(eps) for eps in tqdm(dom2_lin)], dtype=np.float_)
    np.savez("dom2_lin", x=dom2_lin, dos=dos_dom2_lin[:, 0], err=dos_dom2_lin[:, 1])


min_delta = 10

res1_che = np.load("dom1_che.npz")
res1_lin = np.load("dom1_lin.npz")

# make sure the results are accurate enough
assert res1_che['err'].max() < 1e-16
assert res1_lin['err'].max() < 1e-16

dom1 = np.concatenate([res1_che['x'], res1_lin['x']])
# cut points which are too close together
_, relevant = np.unique(np.round(dom1, decimals=min_delta), return_index=True)
dom1 = dom1[relevant]
dos1 = np.concatenate([res1_che['dos'], res1_lin['dos']])[relevant]
sort = np.argsort(dom1)
dom1 = dom1[sort]
dos1 = dos1[sort]


res2_che = np.load("dom2_che.npz")
res2_lin = np.load("dom2_lin.npz")

# make sure the results are accurate enough
assert res2_che['err'].max() < 1e-16
assert res2_lin['err'].max() < 1e-16

dom2 = np.concatenate([res2_che['x'], res2_lin['x']])
# cut points which are too close together
_, relevant = np.unique(np.round(dom2, decimals=min_delta), return_index=True)
dom2 = dom2[relevant]
dos2 = np.concatenate([res2_che['dos'], res2_lin['dos']])[relevant]
sort = np.argsort(dom2)
dom2 = dom2[sort]
dos2 = dos2[sort]


# we have to fix DOS at eps=1/3 per hand, fucked up...
dos1[-1] = np.polynomial.Polynomial.fit(dom1[-20:-1], y=dos1[-20:-1], deg=3)(vanhow)
dos2[0] = dos1[-1]

np.savez("../gftool/lattice/scubic_dos",
         x1=dom1, dos1=dos1, x2=dom2, dos2=dos2)
