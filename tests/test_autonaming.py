import pymc4 as pm
from pymc4.ast_compiler import parse_random_variable_names
import pytest


def model_with_only_good_yields():
    x = yield pm.Normal(loc=0, scale=1)
    y = yield pm.HalfCauchy(scale=2)
    z = yield pm.Binomial(total_count=10, probs=0.2)


def model_with_yields_and_other_assigns():
    N = 10
    N += 10
    x = yield pm.Normal(loc=0, scale=1)
    s: str = "A test string."
    y = yield pm.HalfCauchy(scale=2)
    a, b, c = (1, 2, 3)
    z = yield pm.Binomial(total_count=10, probs=0.2)


def model_with_yield_tuple():
    x, y = yield pm.Normal(loc=0, scale=1), pm.HalfCauchy(scale=2)


def model_with_yield_non_function_call():
    N = yield 10


def test_parsing_model_with_only_good_yields():
    names = parse_random_variable_names(model_with_only_good_yields)
    assert names == ["x", "y", "z"]


def test_parsing_model_with_yields_and_other_assigns():
    names = parse_random_variable_names(model_with_yields_and_other_assigns)
    assert names == ["x", "y", "z"]


def test_parsing_model_with_yield_tuple():
    with pytest.raises(RuntimeError):
        names = parse_random_variable_names(model_with_yield_tuple)


def test_parsing_model_with_yield_non_function_call():
    with pytest.raises(RuntimeError):
        names = parse_random_variable_names(model_with_yield_non_function_call)
