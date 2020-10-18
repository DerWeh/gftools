.. currentmodule:: gftool

What's New
==========

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
* add fitting of high-frequency moment to `gftool.fourier.iw2tau` (`e2c92e2`)

Other New Features
~~~~~~~~~~~~~~~~~~

* add `gftool.pade_frequencies` (`9f492fc`)
* add `gftool.density_iw` function as common interface to calculate occupation
  number from Matsubara or Padé frequencies
* allow calculation of `gftool.lattice.bethe` for Bethe lattice at complex points
  (not that this is probably not a physically meaningful quantity) (`ccbac7b`)
* add stress tensor transformation `gftool.lattice.square.stress_trafo` for 2D (`528fb21`)

Bug fixes
~~~~~~~~~

* Fix constant in `gftool.fourier.tau2iw_ft_lin` (`e2163e3`).
  This error most likely didn't significantly affect any results for a reasonable
  number of tau-points.
* `gftool.density` should work now with gu-style matrices (`4deffdf`)

Documentation
~~~~~~~~~~~~~
* Functions exposed at the top level (`gftool`) should now properly appear in
  the documentation.

0.6.1
-----
