.. GfTool documentation master file, created by
   sphinx-quickstart on Sat Nov 30 17:11:49 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GfTool's documentation!
==================================

:Release: |version|
:Date: |today|
:DOI: `10.5281/zenodo.4744545 <https://doi.org/10.5281/zenodo.4744545>`_

This reference manual details functions, modules, and objects included in `GfTool`,
describing what they are and what they do.

`GfTool` is a collection of commonly used Green's functions and utilities.
The main purpose of this module is to have a tested and thus reliable basis
to do numerics. It happened to me too often, that I just made a mistake copying
the Green's function and was then wondering what was wrong with my algorithm.

.. plot:: dos_gallary.py
   :include-source: False
   :width: 95%
   :alt: Selection of DOSs

   *Selection* of lattice Green's functions or rather the corresponding DOSs available in `GfTool`.

The main use case of `GfTool` was DMFT and its real space generalization,
in particular using CT-QMC algorithms.



.. toctree::
   :includehidden:
   :maxdepth: 3
   :caption: Contents

   getting-started
   tutorial


.. toctree::
   :maxdepth: 1
   :caption: API

   gftool
   generated/gftool.beb
   generated/gftool.cpa
   generated/gftool.fourier
   generated/gftool.lattice
   generated/gftool.matrix
   generated/gftool.pade
   generated/gftool.siam


.. toctree::
   :maxdepth: 1
   :caption: Help

   whats-new



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
