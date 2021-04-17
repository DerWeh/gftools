=======
GfTools
=======

+---------+----------------------+-----------------+
| master  ||build-status-master| ||codecov-master| |
+---------+----------------------+-----------------+
| develop ||build-status-develop|||codecov-develop||
+---------+----------------------+-----------------+


Collection of commonly used Green's functions and utilities.
The main purpose of this module is to have a tested and thus reliable basis
to do numerics. It happened to me too often, that I just made a mistake copying 
the Green's function and was then wondering what was wrong with my algorithm.
The main use case of GfTools was DMFT and its real space generalization,
in particular using CT-QMC algorithms.



Installation
------------

The package is available on PyPi:

.. code-block:: console

   $ pip install gftool

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

Currently the packages main content is

gftool
   * collection of non-interacting Green's functions and spectral functions,
     see also the `lattice` submodule.
   * utility functions like Matsubara frequencies and Fermi functions.
   * reliable calculation of particle numbers via Matsubara sums

fourier
   * Fourier transforms from Matsubara frequencies to imaginary time and back,
     including the handling of high-frequencies moments
     (especially import for transforms from Matsubara to imaginary time)
   * Laplace transform from real times to complex frequencies

matrix
   * helper for Green's functions in matrix form

pade
   * analytic continuation via the Padé algorithm

.. |build-status-master| image:: https://travis-ci.org/DerWeh/gftools.svg?branch=master
   :target: https://travis-ci.org/DerWeh/gftools
   :alt: Build status master
.. |codecov-master| image:: https://codecov.io/gh/DerWeh/gftools/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/DerWeh/gftools
   :alt: Coverage master
.. |build-status-develop| image:: https://travis-ci.org/DerWeh/gftools.svg?branch=develop
   :target: https://travis-ci.org/DerWeh/gftools
   :alt: Build status develop
.. |codecov-develop| image:: https://codecov.io/gh/DerWeh/gftools/branch/develop/graph/badge.svg
   :target: https://codecov.io/gh/DerWeh/gftools
   :alt: Coverage develop
.. _documentation:
   https://derweh.github.io/gftools/
.. _master doc:
   https://gftools.readthedocs.io/en/master/
.. _develop doc:
   https://gftools.readthedocs.io/en/develop/
.. _ReadTheDocs:
   https://gftools.readthedocs.io/en/latest/
.. _pipenv:
   https://pipenv.kennethreitz.org/en/latest/#install-pipenv-today
.. _venv:
   https://docs.python.org/3/library/venv.html
.. _conda:
   https://docs.conda.io/en/latest/
.. _pyenv:
   https://github.com/pyenv/pyenv
