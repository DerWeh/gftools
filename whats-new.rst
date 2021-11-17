.. currentmodule:: gftool

What's New
==========

0.10.1 (not released)
---------------------

New Features
~~~~~~~~~~~~
* Officially support Python 3.10
* Add retarded-time Bethe Green's `~gftool.lattice.bethe.gf_ret_t` function
  (:commit:`6ffc7c91`)

Internal improvements
~~~~~~~~~~~~~~~~~~~~~
* Switch from Travis to GitHub actions #20 (:commit:`23ba0a34`)

  - This adds test for Mac and Windows

Documentation
~~~~~~~~~~~~~
* Fix various errors using `Velin <https://github.com/Carreau/velin>`_
  (:commit:`03ff6c8e`)


Bug fixes
~~~~~~~~~
* Accept singular constrains in `~gftool.linalg.lstsq_ec` (:commit:`167e7886`)


0.10.0 (2021-09-19)
-------------------

Breaking Changes
~~~~~~~~~~~~~~~~
* Drop support for Python 3.6, **minimal version** is now **3.7**
* Content of `gftool.matrix` was renamed more appropriately:

   - `xi` of `~gftool.matrix.Decomposition` is now
     `~gftool.matrix.Decomposition.eig`, as it contains the eigenvalues
   - New functions
     `~gftool.matrix.decompose_mat` for general matrices,
     `~gftool.matrix.decompose_sym` for complex symmetric matrices,
     and `~gftool.matrix.decompose_her` for Hermitian matrices.

Depreciations
~~~~~~~~~~~~~
* Deprecate the `~gftool.matrix` functions
  `~gftool.matrix.decompose_gf`,
  `~gftool.matrix.decompose_hamiltonian`,
  `~gftool.matrix.Decomposition.from_gf`,
  and `~gftool.matrix.Decomposition.from_hamiltonian`.

Documentation
~~~~~~~~~~~~~
* New index page independent of README,
  separated :doc:`getting-started` page.
* Improve :doc:`tutorial` and `gftool.matrix`
* Generate PDF documentation on ReadTheDocs (:commit:`3122e1ba`)

Internal improvements
~~~~~~~~~~~~~~~~~~~~~
* Use eigendecomposition instead of SVD in `gftool.beb` (:commit:`0475c110`)
* Drop slow `~numpy.asfortranarray` in `gftool.matrix` (:commit:`4865cc05`)
* Use `pre-commit` (:commit:`6f4028d3`)



0.9.1 (2021-06-01)
------------------

Bug fixes
~~~~~~~~~
CPA:

* return scalar `mu` in `gftool.cpa.solve_fxdocc_root` (:commit:`10fae4d`)
* find `mu` more reliably

Other New Features
~~~~~~~~~~~~~~~~~~
* SIAM: add greater and lesser Green's functions
  `~gftool.siam.gf0_loc_gr_t` and `~gftool.siam.gf0_loc_le_t` (:commit:`ea541f3`)


0.9.0 (2021-05-09)
------------------

New Features
~~~~~~~~~~~~

Implement `~gftool.cpa` and `~gftool.beb` to treat disorder (:commit:`c3bad20c`)



0.8.1 (2021-04-25)
------------------

New Features
~~~~~~~~~~~~

The 3D cubic lattices were added:

* body-centered cubic `gftool.lattice.bcc` (:commit:`406acef8`)
* face-centered cubic `gftool.lattice.fcc` (:commit:`ddd559cb`)


0.8.0 (2021-04-17)
------------------

New Features
~~~~~~~~~~~~

The `gftool.lattice` module was extended, especially regarding two-dimensional lattice.
There where also some enhancements, given DOS moments are now up to order 20,
and they should be accurate to machine precision.

The following lattices where added with full interface:

* Simple cubic: `gftool.lattice.sc` (:commit:`4e3021`) by `Andreas Östlin <https://github.com/aostlin>`_
* Honeycomb: `gftool.lattice.honeycomb` (:commit:`7aa3133`)
* Triangular: `gftool.lattice.triangular` (:commit:`c56f33e`)

Local Green's function and DOS is now also available for the following lattices:

* Lieb: `gftool.lattice.lieb` (:commit:`c76e948`)
* Kagome: `gftool.lattice.kagome` (:commit:`28a41c0`)
* Bethe lattice with general coordination: `gftool.lattice.bethez` (:commit:`2648cf4`)
* Rectangular: `gftool.lattice.rectangular`

Other New Features
~~~~~~~~~~~~~~~~~~

* add retarded time Green's function give by its poles `gftool.pole_gf_ret_t`
* added `gftool.siam` module with some basics for the non-interacting siam

Depreciations
~~~~~~~~~~~~~

* `gftool.density` is deprecated and will likely be discontinued.
  Consider the more flexible `gftool.density_iw` instead.

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
