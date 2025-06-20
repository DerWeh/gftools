"""Implement additional hypothesis.strategies to generate test data."""
import sys
from functools import partial
from typing import NamedTuple
from collections.abc import Iterable

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import (
    arrays,
    mutually_broadcastable_shapes,
    BroadcastableShapes,
)

class GU_Args(NamedTuple):
    args: "tuple[np.ndarray]"
    shapes:  BroadcastableShapes


if sys.version_info < (3, 10):
    # Inefficient approximate backport! Use only for short iterables.

    def zip_strict(*iterables):
        # consume iterables
        collected = [list(it) for it in iterables]
        if len(set(len(it) for it in collected)) > 1:
            raise ValueError("zip_strict() got arguments of different length")
        return zip(*collected)

else:
    zip_strict = partial(zip, strict=True)


@st.composite
def gufunc_args(draw, shape_kwds: dict, **array_kdws):
    """
    Return a strategy to generate tuple of `numpy` arrays to test broadcasting.

    The returned strategy generates GU_Args.

    Parameters
    ----------
    shape_kwds : dict
        Keyword arguments passed to `mutually_broadcastable_shapes`.
    array_kdws
        Keyword arguments passed to `arrays`.
        If a single value is given, it is used for all arrays.
        Alternatively an iterable with one value per array can be provided.

    """
    shapes = draw(mutually_broadcastable_shapes(**shape_kwds))
    num_arrays = len(shapes.input_shapes)
    kwds_list = [{"shape": shape} for shape in shapes.input_shapes]
    for key, val in array_kdws.items():
        if not isinstance(val, Iterable):
            val = [val]*num_arrays
        for kwds, vi in zip_strict(kwds_list, val):
            kwds[key] = vi

    args = tuple(draw(arrays(**kwds)) for kwds in kwds_list)
    return GU_Args(args, shapes=shapes)
