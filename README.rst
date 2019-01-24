=======
gftools
=======

+---------+----------------------+-----------------+
| master  ||build-status-master| ||codecov-master| |
+---------+----------------------+-----------------+
| develop ||build-status-develop|||codecov-develop||
+---------+----------------------+-----------------+


Collection of commonly used Green's functions and utilities.



Documentation
-------------

The documentation and API can be found here: `documentation`_.

Currently the packages main content is

gftools
   * collection of non-interacting Green's functions and spectral functions
   * utility functions like Matsubara frequencies and Fermi functions.
   * reliable calculation of particle numbers via Matsubara sums

matrix
   * helper for Green's functions in matrix form

pade
   * analytic continuation via the Pade algorithm

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
