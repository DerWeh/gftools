"""Provide top level module for tests."""
from __future__ import absolute_import

import os

from sys import path
from functools import lru_cache

PATH = os.path.abspath(os.path.dirname(__file__))
path.insert(0, os.path.join(PATH, os.pardir, os.pardir))

import gftool
import gftool.beb
import gftool.cpa
import gftool.matrix
import gftool.fourier
import gftool.pade as gt_pade
import gftool.linalg
import gftool.basis.pole as pole
import gftool.siam
import gftool._util


gftool.statistics._pade_frequencies = lru_cache(maxsize=10)(gftool.statistics._pade_frequencies)
