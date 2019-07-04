import pytest
import math
import numpy as np
import random
import scipy.stats as sts
from pymc4 import distributions as dist
from numpy.testing import assert_almost_equal

def test_normal():
    x = random.uniform(-3, 3)
    norm = dist.Normal("n", 0, 1)
    assert_almost_equal(norm.log_prob_numpy(x), sts.norm.logpdf(x))


