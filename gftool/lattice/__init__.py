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

"""
from . import bethe, onedim, square, rectangular, triangular, honeycomb

# silence warnings of unused imports
assert bethe
assert onedim
assert square
assert rectangular
assert triangular
assert honeycomb
