"""Helper functions for tests."""
import numpy as np


def assert_allclose_vm(actual, desired, rtol=1e-7, atol=1e-14, **kwds):
    """Relax `assert_allclose` in case some elements are huge and others 0."""
    # TODO: we should provide an axis argument and somehow iterate...
    # for now we just relax the test
    fact = np.maximum(np.linalg.norm(desired), 1.0)
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol*fact, **kwds)
