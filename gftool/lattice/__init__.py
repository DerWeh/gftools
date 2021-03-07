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

"""
from . import bethe, onedim, square, rectangular, triangular

# silence warnings of unused imports
assert bethe
assert onedim
assert square
assert rectangular
assert triangular
