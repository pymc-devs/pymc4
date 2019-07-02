import pymc4 as pm
import numpy as np
from pymc4 import distributions as dist


def test_simple_model():
    def simple_model():
        norm = yield dist.Normal("n", 0, 1)
        return norm

    _, state = pm.evaluate_model(simple_model())
    assert "n" in state.values


def test_complex_model():
    @pm.model
    def nested_model(cond):
        norm = yield dist.Normal("n", cond, 1)
        return norm

    @pm.model
    def complex_model():
        norm = yield dist.Normal("n", 0, 1)
        result = yield nested_model(norm, name="a")
        return result

    _, state = pm.evaluate_model(complex_model())

    assert set(state.values) == {"complex_model/n", "complex_model/a", "complex_model/a/n", "complex_model"}


def test_complex_model_no_keep_return():
    @pm.model
    def nested_model(cond):
        norm = yield dist.Normal("n", cond, 1)
        return norm

    @pm.model(keep_return=False)
    def complex_model():
        norm = yield dist.Normal("n", 0, 1)
        result = yield nested_model(norm, name="a")
        return result

    _, state = pm.evaluate_model(complex_model())

    assert set(state.values) == {"complex_model/n", "complex_model/a", "complex_model/a/n"}


def test_transformed_model_untransformed_executor():
    def transformed_model():
        norm = yield dist.HalfNormal("n", 1, transform=dist.transforms.Log())
        return norm

    _, state = pm.evaluate_model(transformed_model())

    assert set(state.values) == {"n"}


def test_transformed_model_transformed_executor():
    def transformed_model():
        norm = yield dist.HalfNormal("n", 1, transform=dist.transforms.Log())
        return norm

    _, state = pm.evaluate_model_transformed(transformed_model())

    assert set(state.values) == {"n", "__log_n"}
    assert np.allclose(state.values["n"], np.exp(state.values["__log_n"]))


def test_transformed_model_transformed_executor_with_passed_value():
    def transformed_model():
        norm = yield dist.HalfNormal("n", 1, transform=dist.transforms.Log())
        return norm

    _, state = pm.evaluate_model_transformed(transformed_model(), n=1.)

    assert set(state.values) == {"n", "__log_n"}
    np.testing.assert_allclose(state.values["__log_n"], 0.)

    _, state = pm.evaluate_model_transformed(transformed_model(), __log_n=0.)

    assert set(state.values) == {"n", "__log_n"}
    np.testing.assert_allclose(state.values["n"], 1.)
