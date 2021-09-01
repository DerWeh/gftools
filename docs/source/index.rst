.. GfTool documentation master file, created by
   sphinx-quickstart on Sat Nov 30 17:11:49 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GfTool's documentation!
==================================

:Release: |version|
:Date: |today|

|pypi| |conda-forge| |DOI|

This reference manual details functions, modules, and objects included in `GfTool`,
describing what they are and what they do.

.. contents::  Contents
   :local:

`GfTool` is a collection of commonly used Green's functions and utilities.
The main purpose of this module is to have a tested and thus reliable basis
to do numerics. It happened to me too often, that I just made a mistake copying
the Green's function and was then wondering what was wrong with my algorithm.

.. plot:: dos_gallary.py
   :include-source: False
   :width: 800
   :alt: Selection of DOSs

   *Selection* of lattice Green's functions or rather the corresponding DOSs available in `GfTool`.

The main use case of `GfTool` was DMFT and its real space generalization,
in particular using CT-QMC algorithms.


.. |pypi| image:: https://badge.fury.io/py/gftool.svg
   :target: https://badge.fury.io/py/gftool
   :alt: PyPI release
.. |conda-forge| image:: https://img.shields.io/conda/vn/conda-forge/gftool.svg
   :target: https://anaconda.org/conda-forge/gftool
   :alt: conda-forge release
.. |DOI| image:: https://zenodo.org/badge/115784231.svg
   :target: https://zenodo.org/badge/latestdoi/115784231
   :alt: DOI


GfTool's main content
---------------------

`gftool`
   * collection of non-interacting Green's functions and spectral functions,
     see also the `~gftool.lattice` submodule.
   * utility functions like Matsubara frequencies and Fermi functions.
   * reliable calculation of particle numbers via Matsubara sums

`~gftool.cpa`/`~gftool.beb`
   * Single site approximation to disorder
   * diagonal disorder only (CPA) and diagonal and off-diagonal (BEB)
   * average local Green's function and component Green's functions

`~gftool.fourier`
   * Fourier transforms from Matsubara frequencies to imaginary time and back,
     including the handling of high-frequencies moments
     (especially important for transforms from Matsubara to imaginary time)
   * Laplace transform from real times to complex frequencies

`~gftool.matrix`
   * helper for Green's functions in matrix form

`~gftool.pade`
   * analytic continuation via the Padé algorithm


Installation
------------
The package is available on PyPI:

.. code-block:: console

   $ pip install gftool

Alternatively, it can be installed via GitHub. You can install it using

.. code-block:: console

   $ pip install https://github.com/DerWeh/gftools/archive/VERSION.zip

where `VERSION` can be a release (e.g. `0.5.1`) or a branch (e.g. `develop`).
(As always, it is not advised to install it into your system Python,
consider using `pipenv`_, `venv`_, `conda`_, `pyenv`_, or similar tools.)
Of course you can also clone or fork the project.

For `conda`_ users, `GfTool` is also available on `conda-forge`_

.. code-block:: console

   $ conda install -c conda-forge gftool

If you clone the project, you can locally build the documentation:

.. code-block:: console

   $ pip install -r requirements-doc.txt
   $ python setup.py build_sphinx


.. _pipenv:
   https://pipenv.kennethreitz.org/en/latest/#install-pipenv-today
.. _venv:
   https://docs.python.org/3/library/venv.html
.. _conda:
   https://docs.conda.io/en/latest/
.. _conda-forge:
   https://anaconda.org/conda-forge/gftool
.. _pyenv:
   https://github.com/pyenv/pyenv



Note on documentation
---------------------
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


.. toctree::
   :includehidden:
   :maxdepth: 3
   :caption: Contents
   :hidden:

   self
   tutorial


.. toctree::
   :maxdepth: 2
   :caption: API
   :hidden:

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
   :hidden:

   whats-new



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
