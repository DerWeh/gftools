"""Collection of different lattices and their Green's functions.

Submodules
----------

.. autosummary::
    :toctree:  _autosummary

    bethe
    square

"""
from . import bethe, square

# silence warnings of unused imports
assert bethe
assert square
