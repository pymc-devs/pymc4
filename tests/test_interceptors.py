from tensorflow_probability import edward2 as ed
import tensorflow as tf
import numpy as np
import pytest
import pymc4 as pm
from pymc4.util.interceptors import *
# pylint: disable=unused-variable, unused-argument
# pylint: disable-msg=E0102


def test_interceptor():
    interceptor = Interceptor()
