"""Provide top level module for tests."""
import os

from sys import path
from functools import lru_cache

import numpy as np

PATH = os.path.abspath(os.path.dirname(__file__))
path.insert(0, os.path.join(PATH, os.pardir, os.pardir))

import gftool
import gftool.beb
import gftool.cpa
import gftool.hermpade
import gftool.matrix
import gftool.fourier
import gftool.polepade
import gftool.pade as gt_pade
import gftool.linalg
import gftool.basis.pole as pole
import gftool.siam
import gftool._util


gftool.statistics._pade_frequencies = lru_cache(maxsize=10)(gftool.statistics._pade_frequencies)


def assert_allclose_vm(actual, desired, rtol=1e-7, atol=1e-14, **kwds):
    """Relax `assert_allclose` in case some elements are huge and others 0."""
    # TODO: we should provide an axis argument and somehow iterate...
    # for now we just relax the test
    fact = np.maximum(np.linalg.norm(desired), 1.0)
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol*fact, **kwds)
