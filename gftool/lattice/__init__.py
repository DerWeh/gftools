"""Collection of different lattices and their Green's functions.

Submodules
----------

.. autosummary::
    :toctree:

    bethe
    bethez
    onedim
    square
    rectangular
    lieb
    triangular
    honeycomb
    kagome
    sc
    bcc
    fcc

"""
from . import (bcc, bethe, bethez, fcc, honeycomb, kagome, lieb, onedim,
               rectangular, sc, square, triangular)

# silence warnings of unused imports
assert bcc
assert bethe
assert bethez
assert fcc
assert honeycomb
assert kagome
assert lieb
assert onedim
assert rectangular
assert sc
assert square
assert triangular
