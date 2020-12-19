from math import fsum

import numpy as np

from numpy.random import default_rng

from .context import gftool as gt


def test_gu_sum_accuracy():
    """Test gu-sum accuracy for Fortran and C order against accurate `fsum`."""
    rng = default_rng(0)
    test_data = rng.random((2, 10000))
    test_c = np.ascontiguousarray(test_data)
    test_f = np.asfortranarray(test_data)
    assert np.all(gt._util._gu_sum(test_c) == gt._util._gu_sum(test_f))
    gu_sum = gt._util._gu_sum(test_data)
    c_sum = np.sum(test_c, axis=-1)
    f_sum = np.sum(test_f, axis=-1)
    reference = [fsum(test_row) for test_row in test_data]
    assert np.all(abs(gu_sum - reference) <= abs(c_sum - reference))
    assert np.all(abs(gu_sum - reference) <= abs(f_sum - reference))
