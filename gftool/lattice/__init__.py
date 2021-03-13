"""Collection of different lattices and their Green's functions.

Submodules
----------

.. autosummary::
    :toctree:

    bethe
    onedim
    square
    rectangular
    triangular
    honeycomb
    sc

"""
from . import (bethe, honeycomb, onedim, rectangular, sc, square,
               triangular)

# silence warnings of unused imports
assert bethe
assert onedim
assert square
assert rectangular
assert triangular
assert honeycomb
assert sc
