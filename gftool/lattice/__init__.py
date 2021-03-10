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
    simplecubic

"""
from . import (bethe, honeycomb, onedim, rectangular, simplecubic, square,
               triangular)

# silence warnings of unused imports
assert bethe
assert onedim
assert square
assert rectangular
assert triangular
assert honeycomb
assert simplecubic
