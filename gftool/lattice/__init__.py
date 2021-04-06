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
    triangular
    honeycomb
    kagome
    sc

"""
from . import (bethe, bethez, honeycomb, kagome, onedim, rectangular, sc,
               square, triangular)

# silence warnings of unused imports
assert bethe
assert bethez
assert honeycomb
assert kagome
assert onedim
assert rectangular
assert sc
assert square
assert triangular
