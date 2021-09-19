======
GfTool
======

+---------+----------------------+-----------------+-------------+
| master  ||build-status-master| ||codecov-master| ||rtd-master| |
+---------+----------------------+-----------------+-------------+
| develop ||build-status-develop|||codecov-develop|||rtd-develop||
+---------+----------------------+-----------------+-------------+

|pypi| |conda-forge| |DOI| |lgtm|

Collection of commonly used Green's functions and utilities.
The main purpose of this module is to have a tested and thus reliable basis
to do numerics. It happened to me too often, that I just made a mistake copying
the Green's function and was then wondering what was wrong with my algorithm.
For example, a *selection* of lattice Green's functions or rather the corresponding DOSs:

.. image:: https://gftools.readthedocs.io/en/develop/_images/dos_gallary.png
   :width: 800
   :alt: Selection of DOSs

Also have a look at the `tutorial`_ to get an idea.

The main use case of `GfTool` was DMFT and its real space generalization,
in particular using CT-QMC algorithms.



Installation
------------

The package is available on PyPI_:

.. code-block:: console

   $ pip install gftool

For `conda`_ users, `GfTool` is also available on `conda-forge`_

.. code-block:: console

   $ conda install -c conda-forge gftool

Alternatively, it can be installed via GitHub. You can install it using

.. code-block:: console

   $ pip install https://github.com/DerWeh/gftools/archive/VERSION.zip

where `VERSION` can be a release (e.g. `0.5.1`) or a branch (e.g. `develop`).
(As always, it is not advised to install it into your system Python,
consider using `pipenv`_, `venv`_, `conda`_, `pyenv`_, or similar tools.)
Of course you can also clone or fork the project.

If you clone the project, you can locally build the documentation:

.. code-block:: console

   $ pip install -r requirements-doc.txt
   $ python setup.py build_sphinx



Documentation
-------------

The documentation and API is on `ReadTheDocs`_.
The documentation of specific branches can also be accessed:
`master doc`_, `develop doc`_.
There is also a GitHub page: `documentation`_.

Currently, the package's main content is

gftool
   * collection of non-interacting Green's functions and spectral functions,
     see also the `lattice` submodule.
   * utility functions like Matsubara frequencies and Fermi functions.
   * reliable calculation of particle numbers via Matsubara sums

cpa/beb
   * Single site approximation to disorder
   * diagonal disorder only (CPA) and diagonal and off-diagonal (BEB)
   * average local Green's function and component Green's functions

fourier
   * Fourier transforms from Matsubara frequencies to imaginary time and back,
     including the handling of high-frequencies moments
     (especially important for transforms from Matsubara to imaginary time)
   * Laplace transform from real times to complex frequencies

matrix
   * helper for Green's functions in matrix form

pade
   * analytic continuation via the Padé algorithm

.. |build-status-master| image:: https://app.travis-ci.com/DerWeh/gftools.svg?branch=master
   :target: https://app.travis-ci.com/DerWeh/gftools
   :alt: Build status master
.. |codecov-master| image:: https://codecov.io/gh/DerWeh/gftools/branch/master/graph/badge.svg
   :target: https://app.codecov.io/gh/DerWeh/gftools/branch/master
   :alt: Coverage master
.. |rtd-master| image:: https://readthedocs.org/projects/gftools/badge/?version=master
   :target: https://gftools.readthedocs.io/en/master/?badge=master
   :alt: Documentation Status master
.. |build-status-develop| image:: https://app.travis-ci.com/DerWeh/gftools.svg?branch=develop
   :target: https://app.travis-ci.com/DerWeh/gftools
   :alt: Build status develop
.. |codecov-develop| image:: https://codecov.io/gh/DerWeh/gftools/branch/develop/graph/badge.svg
   :target: https://app.codecov.io/gh/DerWeh/gftools/branch/develop
   :alt: Coverage develop
.. |rtd-develop| image:: https://readthedocs.org/projects/gftools/badge/?version=develop
   :target: https://gftools.readthedocs.io/en/develop/?badge=develop
   :alt: Documentation Status
.. |pypi| image:: https://badge.fury.io/py/gftool.svg
   :target: https://badge.fury.io/py/gftool
   :alt: PyPI release
.. |conda-forge| image:: https://img.shields.io/conda/vn/conda-forge/gftool.svg
   :target: https://anaconda.org/conda-forge/gftool
   :alt: conda-forge release
.. |DOI| image:: https://zenodo.org/badge/115784231.svg
   :target: https://zenodo.org/badge/latestdoi/115784231
   :alt: DOI
.. |lgtm| image:: https://img.shields.io/lgtm/grade/python/g/DerWeh/gftools.svg?logo=lgtm&logoWidth=18
   :target: https://lgtm.com/projects/g/DerWeh/gftools/context:python
   :alt: Code quality
.. _documentation:
   https://derweh.github.io/gftools/
.. _master doc:
   https://gftools.readthedocs.io/en/master/
.. _develop doc:
   https://gftools.readthedocs.io/en/develop/
.. _ReadTheDocs:
   https://gftools.readthedocs.io/en/latest/
.. _tutorial:
   https://gftools.readthedocs.io/en/develop/tutorial.html
.. _PyPi:
   https://pypi.org/project/gftool/
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
