.. currentmodule:: gftool

What's New
==========


Not Released
------------

New Features
~~~~~~~~~~~~

* add Laplace transformation from real times to complex frequencies `gftool.fourier.tt2z`

Other New Features
~~~~~~~~~~~~~~~~~~

* add retarded time Green's function give by its poles `gftool.pole_gf_ret_t`

Documentation
~~~~~~~~~~~~~

* Button to toggle the prompt (>>>) was added (:commit:`46b6f39`)

Internal improvements
~~~~~~~~~~~~~~~~~~~~~

* Ensure more accurate `numpy.sum` using partial pairwise summation for 
  generalized ufuncs (:commit:`2d3baef`)



0.7.0 (2020-10-18)
------------------

Breaking Changes
~~~~~~~~~~~~~~~~

* The `gftool.pade` module had a minor rework.
  The behavior of filters changed. Future breaking changes are to be expected,
  the module is not well structured.
 
New Features
~~~~~~~~~~~~

* add `gftool.lattice.onedim` for Green's function of one-dimensional lattice
* add fitting of high-frequency moment to `gftool.fourier.iw2tau` (:commit:`e2c92e2`)

Other New Features
~~~~~~~~~~~~~~~~~~

* add `gftool.pade_frequencies` (:commit:`9f492fc`)
* add `gftool.density_iw` function as common interface to calculate occupation
  number from Matsubara or Padé frequencies
* allow calculation of `gftool.lattice.bethe` for Bethe lattice at complex points
  (note, that this is probably not a physically meaningful quantity) (:commit:`ccbac7b`)
* add stress tensor transformation `gftool.lattice.square.stress_trafo` for 2D (:commit:`528fb21`)

Bug fixes
~~~~~~~~~

* Fix constant in `gftool.fourier.tau2iw_ft_lin` (:commit:`e2163e3`).
  This error most likely didn't significantly affect any results for a reasonable
  number of tau-points.
* `gftool.density` should work now with gu-style matrices (:commit:`4deffdf`)

Documentation
~~~~~~~~~~~~~
* Functions exposed at the top level (`gftool`) should now properly appear in
  the documentation.



0.6.1
-----
