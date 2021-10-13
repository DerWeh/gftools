Getting started
===============


GfTool's main content
---------------------

`gftool`
   * collection of non-interacting Green's functions and spectral functions,
     see also the `~gftool.lattice` submodule.
   * utility functions like Matsubara frequencies and Fermi functions.
   * reliable calculation of particle numbers via Matsubara sums

`~gftool.cpa`/`~gftool.beb`
   * single site approximation to disorder
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
