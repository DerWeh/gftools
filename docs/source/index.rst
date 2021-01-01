.. GfTool documentation master file, created by
   sphinx-quickstart on Sat Nov 30 17:11:49 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GfTool's documentation!
===================================

:Release: |version|
:Date: |today|

This reference manual details functions, modules, and objects included in GfTools,
describing what they are and what they do.

.. toctree::
   :maxdepth: 2
   :caption: Contents:


README
======

.. include:: ../../README.rst


Note on documentation
=====================
We try to follow `numpy` broadcasting rules. Many functions acting on an axis
act like generalized `ufuncs`_. In this case, a function can be called for
stacked arguments instead of looping over the specific arguments.

We indicate this by argument shapes containing an ellipse e.g. `(...)` or `(..., N)`.
It must be possible for all ellipses to be broadcasted against each other.
A good example is the `~gftool.fourier` module.

We calculate the Fourier transforms `~gftool.fourier.iw2tau` for Green's 
functions with different on-site energies without looping:

>>> e_onsite = np.array([-0.5, 0, 0.5])
>>> beta = 10
>>> iws = gt.matsubara_frequencies(range(1024), beta=beta)
>>> gf_iw = gt.bethe_gf_z(iws - e_onsite[..., np.newaxis], half_bandwidth=1.0)
>>> gf_iw.shape
(3, 1024)

>>> from gftool import fourier
>>> gf_tau = fourier.iw2tau(gf_iw, beta=beta, moments=np.ones([1]))
>>> gf_tau.shape
(3, 2049)

The moments are automatically broadcasted.
We can also explicitly give the second moments:

>>> moments = np.stack([np.ones([3]), e_onsite], axis=-1)
>>> gf_tau = fourier.iw2tau(gf_iw, beta=beta, moments=moments)
>>> gf_tau.shape
(3, 2049)


.. _ufuncs: https://numpy.org/doc/stable/reference/ufuncs.html


API
===


.. autosummary::
  :toctree: generated

  gftool
  gftool.fourier
  gftool.lattice
  gftool.matrix
  gftool.pade
  gftool.siam


.. toctree::
   :maxdepth: 1
   :caption: Help

   whats-new



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
