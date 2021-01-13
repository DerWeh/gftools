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
    sc

"""
from . import (bethe, bethez, honeycomb, onedim, rectangular, sc, square,
               triangular)

# silence warnings of unused imports
assert bethe
assert bethez
assert onedim
assert square
assert rectangular
assert triangular
assert honeycomb
assert sc
