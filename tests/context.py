"""Provide toplevel moduel for tests"""
from __future__ import absolute_import
import os

from sys import path

PATH = os.path.abspath(os.path.dirname(__file__))
path.insert(0, os.path.join(PATH, os.pardir))

import __init__ as gftools
