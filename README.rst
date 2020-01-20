=======
GfTools
=======

+---------+----------------------+-----------------+
| master  ||build-status-master| ||codecov-master| |
+---------+----------------------+-----------------+
| develop ||build-status-develop|||codecov-develop||
+---------+----------------------+-----------------+


Collection of commonly used Green's functions and utilities.
The main propose of this module is to have a tested and thus reliable basis
to do numerics. It happened to me too often, that I just made a mistake copying 
the Green's function and was then wondering what was wrong with my algorithm.
The main use case of GfTools was DMFT and its real space generalization,
in particular using CT-QMC algorithms.



Installation
------------

The package is available on PyPi:

.. code-block:: console

   # pip install gftool

Alternatively, it can be installed via GitHub. You can install it using

.. code-block:: console

   $ pip install https://github.com/DerWeh/gftools/archive/VERSION.zip

where `VERSION` can be a release (e.g. `0.5.1`) or a branch (e.g. `develop`).
(As always, it is not advised to install it into your system Python,
consider using `pipenv`_, `venv`_, `conda`_, `pyenv`_, or similar tools.)
Of course you can also clone or fork the project.



Documentation
-------------

The documentation and API can be found here: `documentation`_.
There is now also documentation on ReadTheDocs:
`master doc`_, `develop doc`_, `latest doc`_

Currently the packages main content is

gftool
   * collection of non-interacting Green's functions and spectral functions
     see the `lattice` submodule
   * utility functions like Matsubara frequencies and Fermi functions.
   * reliable calculation of particle numbers via Matsubara sums
     (Needs a refactor and more accurate extrapolation)

fourier
   * Fourier transforms from Matsubara frequencies to imaginary time and back
     Handling of high-frequencies moments is not yet included and has to be
     done by hand (especially import for transforms from Matsubara to imaginary
     time)

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
.. _latest doc:
   https://gftools.readthedocs.io/en/latest/
.. _pipenv:
   https://pipenv.kennethreitz.org/en/latest/#install-pipenv-today
.. _venv:
   https://docs.python.org/3/library/venv.html
.. _conda:
   https://docs.conda.io/en/latest/
.. _pyenv:
   https://github.com/pyenv/pyenv
