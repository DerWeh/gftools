"""Collection of different lattices and their Green's functions.

Submodules
----------

.. autosummary::
    :toctree:

    bethe
    onedim
    square

"""
from . import bethe, onedim, square

# silence warnings of unused imports
assert bethe
assert onedim
assert square
