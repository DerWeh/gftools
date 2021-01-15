"""Collection of different lattices and their Green's functions.

Submodules
----------

.. autosummary::
    :toctree:

    bethe
    onedim
    scubic
    square

"""
from . import bethe, onedim, scubic, square

# silence warnings of unused imports
assert bethe
assert onedim
assert scubic
assert square
