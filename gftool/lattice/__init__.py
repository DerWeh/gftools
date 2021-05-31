r"""Collection of different lattices and their Green's functions.

The lattices are described by a tight binding Hamiltonian

.. math:: H = t ∑_{⟨i,j⟩ σ} c^†_{iσ} c_{jσ},

where :math:`t` is the hopping amplitude or integral.
Mind the sign, often tight binding Hamiltonians are instead defined with a
negative sign in front of :math:`t`.

The Hamiltonian can be diagonalized

.. math:: H = ∑_{kσ} ϵ_{k} c^†_{kσ} c_{kσ}.

Typical quantities provided for the different lattices are:

:`gf_z`: The one-particle Green's function

   .. math:: G_{ii}(z) = ⟨⟨c_{iσ}|c^†_{iσ}⟩⟩(z) = 1/N ∑_k \frac{1}{z - ϵ_k}.

:`dos`: The density of states (DOS)

   .. math:: DOS(ϵ) = 1/N ∑_k δ(ϵ - ϵₖ).

:`dos_moment`: The moments of the DOS

    .. math:: ϵ^{(m)} = ∫dϵ DOS(ϵ) ϵ^m


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
