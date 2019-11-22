import pytest
import pymc4 as pm
import math
import numpy as np
from pymc4 import distributions as dist
import tensorflow as tf


@pytest.fixture("module")
def complex_model():
    @pm.model
    def nested_model(cond):
        norm = yield dist.HalfNormal("n", cond ** 2, transform=dist.transforms.Log())
        return norm

    @pm.model(keep_return=False)
    def complex_model():
        norm = yield dist.Normal("n", 0, 1)
        result = yield nested_model(norm, name="a")
        return result

    return complex_model


@pytest.fixture("module")
def complex_model_with_observed():
    @pm.model
    def nested_model(cond):
        norm = yield dist.HalfNormal(
            "n", cond ** 2, observed=np.ones(10), transform=dist.transforms.Log()
        )
        return norm

    @pm.model(keep_return=False)
    def complex_model():
        norm = yield dist.Normal("n", 0, 1)
        result = yield nested_model(norm, name="a")
        return result

    return complex_model


@pytest.fixture("module")
def simple_model():
    def simple_model():
        norm = yield dist.Normal("n", 0, 1)
        return norm

    return simple_model


@pytest.fixture("module")
def transformed_model():
    def transformed_model():
        norm = yield dist.HalfNormal("n", 1, transform=dist.transforms.Log())
        return norm

    return transformed_model


@pytest.fixture("module")
def transformed_model_with_observed():
    def transformed_model_with_observed():
        norm = yield dist.HalfNormal("n", 1, transform=dist.transforms.Log(), observed=1.0)
        return norm

    return transformed_model_with_observed


@pytest.fixture("module")
def class_model():
    class ClassModel:
        @pm.model(method=True)
        def class_model_method(self):
            norm = yield pm.Normal("n", 0, 1)
            return norm

    return ClassModel()


@pytest.fixture("module")
def model_with_deterministics():
    expected_deterministics = ["model/abs_norm", "model/sine_norm", "model/norm_copy"]
    expected_ops = [np.abs, np.sin, lambda x: x]
    expected_ops_inputs = [["model/norm"], ["model/norm"], ["model/norm"]]

    @pm.model
    def model():
        norm = yield dist.Normal("norm", 0, 1)
        abs_norm = yield dist.Deterministic("abs_norm", tf.abs(norm))
        sine_norm = yield dist.Deterministic("sine_norm", tf.sin(norm))
        norm_copy = yield dist.Deterministic("norm_copy", norm)
        obs = yield dist.Normal("obs", 0, abs_norm)

    return model, expected_deterministics, expected_ops, expected_ops_inputs


@pytest.fixture(scope="function")
def deterministics_in_nested_models():
    @pm.model
    def nested_model(cond):
        x = yield pm.Normal("x", cond, 1)
        dx = yield pm.Deterministic("dx", x + 1)
        return dx

    @pm.model
    def outer_model():
        cond = yield pm.HalfNormal("cond", 1)
        dcond = yield pm.Deterministic("dcond", cond * 2)
        dx = yield nested_model(dcond)
        ddx = yield pm.Deterministic("ddx", dx)
        return ddx

    expected_untransformed = {
        "outer_model/cond",
        "outer_model/nested_model/x",
    }
    expected_transformed = {"outer_model/__log_cond"}
    expected_deterministics = {
        "outer_model/dcond",
        "outer_model/ddx",
        "outer_model/nested_model/dx",
        "outer_model/nested_model",
        "outer_model",
    }
    deterministic_mapping = {
        "outer_model/dcond": (["outer_model/cond"], lambda x: x * 2),
        "outer_model/ddx": (["outer_model/nested_model/x"], lambda x: x + 1),
        "outer_model/nested_model/dx": (["outer_model/nested_model/x"], lambda x: x + 1),
        "outer_model/nested_model/dx": (["outer_model/nested_model/x"], lambda x: x + 1),
        "outer_model/nested_model": (["outer_model/nested_model/x"], lambda x: x + 1),
        "outer_model": (["outer_model/nested_model/x"], lambda x: x + 1),
    }

    return (
        outer_model,
        expected_untransformed,
        expected_transformed,
        expected_deterministics,
        deterministic_mapping,
    )


def test_class_model(class_model):
    """Test that model can be defined as method in an object definition"""
    _, state = pm.evaluate_model(class_model.class_model_method())
    assert "class_model_method/n" in state.untransformed_values
    assert not state.observed_values
    assert not state.transformed_values


def test_simple_model(simple_model):
    _, state = pm.evaluate_model(simple_model())
    assert "n" in state.untransformed_values
    assert not state.observed_values
    assert not state.transformed_values


def test_complex_model_keep_return():
    @pm.model
    def nested_model(cond):
        norm = yield dist.HalfNormal("n", cond ** 2, transform=dist.transforms.Log())
        return norm

    @pm.model()
    def complex_model():
        norm = yield dist.Normal("n", 0, 1)
        result = yield nested_model(norm, name="a")
        return result

    _, state = pm.evaluate_model(complex_model())

    assert set(state.untransformed_values) == {"complex_model/n", "complex_model/a/n"}
    assert set(state.deterministics) == {
        "complex_model",
        "complex_model/a",
    }
    assert not state.transformed_values  # we call untransformed executor
    assert not state.observed_values


def test_complex_model_no_keep_return(complex_model):
    _, state = pm.evaluate_model(complex_model())

    assert set(state.untransformed_values) == {
        "complex_model/n",
        "complex_model/a/n",
    }
    assert set(state.deterministics) == {"complex_model/a"}
    assert not state.transformed_values  # we call untransformed executor
    assert not state.observed_values


def test_transformed_model_untransformed_executor(transformed_model):
    _, state = pm.evaluate_model(transformed_model())

    assert set(state.untransformed_values) == {"n"}
    assert not state.transformed_values  # we call untransformed executor
    assert not state.observed_values


def test_transformed_model_transformed_executor(transformed_model):
    _, state = pm.evaluate_model_transformed(transformed_model())

    assert set(state.all_values) == {"n", "__log_n"}
    assert set(state.transformed_values) == {"__log_n"}
    assert set(state.untransformed_values) == {"n"}
    assert not state.observed_values
    assert np.allclose(
        state.untransformed_values["n"], math.exp(state.transformed_values["__log_n"])
    )


def test_transformed_model_transformed_executor_with_passed_value(transformed_model):
    _, state = pm.evaluate_model_transformed(transformed_model(), values=dict(n=1.0))

    assert set(state.all_values) == {"n", "__log_n"}
    assert set(state.transformed_values) == {"__log_n"}
    assert set(state.untransformed_values) == {"n"}
    np.testing.assert_allclose(state.all_values["__log_n"], 0.0)

    _, state = pm.evaluate_model_transformed(transformed_model(), values=dict(__log_n=0.0))

    assert set(state.all_values) == {"n", "__log_n"}
    assert set(state.transformed_values) == {"__log_n"}
    assert set(state.untransformed_values) == {"n"}
    np.testing.assert_allclose(state.all_values["n"], 1.0)


def test_transformed_executor_logp_tensorflow(transformed_model):
    tfp = pytest.importorskip("tensorflow_probability")
    bij = tfp.bijectors
    tfd = tfp.distributions

    norm_log = tfd.TransformedDistribution(tfd.HalfNormal(1), bij.Invert(bij.Exp()))

    _, state = pm.evaluate_model_transformed(transformed_model(), values=dict(__log_n=-math.pi))
    np.testing.assert_allclose(
        state.collect_log_prob(), norm_log.log_prob(-math.pi), equal_nan=False
    )

    _, state = pm.evaluate_model_transformed(transformed_model(), values=dict(n=math.exp(-math.pi)))
    np.testing.assert_allclose(
        state.collect_log_prob(), norm_log.log_prob(-math.pi), equal_nan=False
    )


def test_executor_logp_tensorflow(transformed_model):
    tfp = pytest.importorskip("tensorflow_probability")
    tfd = tfp.distributions

    norm = tfd.HalfNormal(1)

    _, state = pm.evaluate_model(transformed_model(), values=dict(n=math.pi))

    np.testing.assert_allclose(state.collect_log_prob(), norm.log_prob(math.pi), equal_nan=False)


def test_single_distribution():
    _, state = pm.evaluate_model(pm.distributions.Normal("n", 0, 1))
    assert "n" in state.all_values


def test_raise_if_return_distribution():
    def invalid_model():
        yield pm.distributions.Normal("n1", 0, 1)
        return pm.distributions.Normal("n2", 0, 1)

    with pytest.raises(pm.flow.executor.EvaluationError) as e:
        pm.evaluate_model(invalid_model())
    assert e.match("should not contain")


def test_observed_are_passed_correctly(complex_model_with_observed):
    _, state = pm.evaluate_model(complex_model_with_observed())

    assert set(state.untransformed_values) == {"complex_model/n"}
    assert set(state.deterministics) == {"complex_model/a"}
    assert not state.transformed_values  # we call untransformed executor
    assert set(state.observed_values) == {"complex_model/a/n"}
    assert np.allclose(state.all_values["complex_model/a/n"], np.ones(10))


def test_observed_are_set_to_none_for_posterior_predictive_correctly(complex_model_with_observed):
    _, state = pm.evaluate_model(
        complex_model_with_observed(), observed={"complex_model/a/n": None}
    )

    assert set(state.untransformed_values) == {
        "complex_model/n",
        "complex_model/a/n",
    }
    assert set(state.deterministics) == {"complex_model/a"}
    assert not state.transformed_values  # we call untransformed executor
    assert not state.observed_values
    assert not np.allclose(state.all_values["complex_model/a/n"], np.ones(10))


def test_observed_do_not_produce_transformed_values(transformed_model_with_observed):
    _, state = pm.evaluate_model_transformed(transformed_model_with_observed())
    assert set(state.observed_values) == {"n"}
    assert not state.transformed_values
    assert not state.untransformed_values


def test_observed_do_not_produce_transformed_values_case_programmatic(transformed_model):
    _, state = pm.evaluate_model_transformed(transformed_model(), observed=dict(n=1.0))
    assert set(state.observed_values) == {"n"}
    assert not state.transformed_values
    assert not state.untransformed_values


def test_observed_do_not_produce_transformed_values_case_override(transformed_model_with_observed):
    _, state = pm.evaluate_model_transformed(
        transformed_model_with_observed(), observed=dict(n=None)
    )
    assert not state.observed_values
    assert set(state.transformed_values) == {"__log_n"}
    assert set(state.untransformed_values) == {"n"}


def test_observed_do_not_produce_transformed_values_case_override_with_set_value(
    transformed_model_with_observed,
):
    _, state = pm.evaluate_model_transformed(
        transformed_model_with_observed(), values=dict(n=1.0), observed=dict(n=None)
    )
    assert not state.observed_values
    assert set(state.transformed_values) == {"__log_n"}
    assert set(state.untransformed_values) == {"n"}
    np.testing.assert_allclose(state.all_values["__log_n"], 0.0)

    _, state = pm.evaluate_model_transformed(
        transformed_model_with_observed(), values=dict(__log_n=0.0), observed=dict(n=None)
    )
    assert not state.observed_values
    assert set(state.transformed_values) == {"__log_n"}
    assert set(state.untransformed_values) == {"n"}
    np.testing.assert_allclose(state.all_values["n"], 1.0)


def test_observed_cant_mix_with_untransformed_and_raises_an_error_case_transformed_executor(
    transformed_model_with_observed,
):
    with pytest.raises(pm.flow.executor.EvaluationError) as e:
        _, state = pm.evaluate_model_transformed(
            transformed_model_with_observed(), values=dict(n=0.0)
        )
    assert e.match("{'n': None}")
    assert e.match("'n' from untransformed values")


def test_observed_cant_mix_with_untransformed_and_raises_an_error_case_untransformed_executor(
    transformed_model_with_observed,
):
    with pytest.raises(pm.flow.executor.EvaluationError) as e:
        _, state = pm.evaluate_model(transformed_model_with_observed(), values=dict(n=0.0))
    assert e.match("{'n': None}")
    assert e.match("'n' from untransformed values")


def test_observed_cant_mix_with_transformed_and_raises_an_error(transformed_model_with_observed):
    with pytest.raises(pm.flow.executor.EvaluationError) as e:
        _, state = pm.evaluate_model_transformed(
            transformed_model_with_observed(), values=dict(__log_n=0.0)
        )
    assert e.match("{'n': None}")
    assert e.match("'__log_n' from transformed values")


def test_as_sampling_state_works_observed_is_constrained(complex_model_with_observed):
    _, state = pm.evaluate_model(complex_model_with_observed())
    sampling_state = state.as_sampling_state()
    assert not sampling_state.transformed_values
    assert set(sampling_state.observed_values) == {"complex_model/a/n"}
    assert set(sampling_state.untransformed_values) == {"complex_model/n"}


def test_as_sampling_state_works_observed_is_set_to_none(complex_model_with_observed):
    _, state = pm.evaluate_model_transformed(
        complex_model_with_observed(), observed={"complex_model/a/n": None}
    )
    sampling_state = state.as_sampling_state()
    assert set(sampling_state.transformed_values) == {"complex_model/a/__log_n"}
    assert not sampling_state.observed_values
    assert set(sampling_state.untransformed_values) == {"complex_model/n"}


def test_as_sampling_state_works_if_transformed_exec(complex_model_with_observed):
    _, state = pm.evaluate_model_transformed(complex_model_with_observed())
    sampling_state = state.as_sampling_state()
    assert not sampling_state.transformed_values
    assert set(sampling_state.observed_values) == {"complex_model/a/n"}
    assert set(sampling_state.untransformed_values) == {"complex_model/n"}


def test_as_sampling_state_does_not_works_if_untransformed_exec(complex_model):
    _, state = pm.evaluate_model(complex_model())
    with pytest.raises(TypeError) as e:
        state.as_sampling_state()
    e.match("'complex_model/a/__log_n' is not found")


def test_unnamed_distribution():
    f = lambda: (yield pm.distributions.Normal.dist(0, 1))
    with pytest.raises(pm.flow.executor.EvaluationError) as e:
        pm.evaluate_model(f())
    assert e.match("anonymous Distribution")


def test_unnamed_distribution_to_prior():
    f = lambda: (yield pm.distributions.Normal.dist(0, 1).prior("n"))
    _, state = pm.evaluate_model(f())
    assert "n" in state.untransformed_values


def test_initialized_distribution_cant_be_transformed_into_a_new_prior():
    with pytest.raises(TypeError) as e:
        pm.distributions.Normal("m", 0, 1).prior("n")
    assert e.match("already not anonymous")


def test_unable_to_create_duplicate_variable():
    def invdalid_model():
        yield pm.distributions.HalfNormal("n", 1, transform=pm.distributions.transforms.Log())
        yield pm.distributions.Normal("n", 0, 1)

    with pytest.raises(pm.flow.executor.EvaluationError) as e:
        pm.evaluate_model(invdalid_model())
    assert e.match("duplicate")
    with pytest.raises(pm.flow.executor.EvaluationError) as e:
        pm.evaluate_model_transformed(invdalid_model())
    assert e.match("duplicate")


def test_unnamed_return():
    @pm.model
    def a_model():
        return (
            yield pm.distributions.HalfNormal("n", 1, transform=pm.distributions.transforms.Log())
        )

    _, state = pm.evaluate_model(a_model())
    assert "a_model" in state.deterministics

    with pytest.raises(pm.flow.executor.EvaluationError) as e:
        pm.evaluate_model(a_model(name=None))
    assert e.match("unnamed")

    with pytest.raises(pm.flow.executor.EvaluationError) as e:
        pm.evaluate_model_transformed(a_model(name=None))
    assert e.match("unnamed")


def test_unnamed_return_2():
    @pm.model(name=None)
    def a_model():
        return (
            yield pm.distributions.HalfNormal("n", 1, transform=pm.distributions.transforms.Log())
        )

    _, state = pm.evaluate_model(a_model(name="b_model"))
    assert "b_model" in state.deterministics

    with pytest.raises(pm.flow.executor.EvaluationError) as e:
        pm.evaluate_model(a_model())
    assert e.match("unnamed")

    with pytest.raises(pm.flow.executor.EvaluationError) as e:
        pm.evaluate_model_transformed(a_model())
    assert e.match("unnamed")


def test_uncatched_exception_works():
    @pm.model
    def a_model():
        try:
            yield 1
        except:
            pass
        yield pm.distributions.HalfNormal("n", 1, transform=pm.distributions.transforms.Log())

    with pytest.raises(pm.flow.executor.StopExecution) as e:
        pm.evaluate_model(a_model())
    assert e.match("something_bad")

    with pytest.raises(pm.flow.executor.StopExecution) as e:
        pm.evaluate_model_transformed(a_model())
    assert e.match("something_bad")


def test_none_yield():
    def model():
        yield None
        yield dist.Normal("n", 0, 1)

    with pytest.raises(pm.flow.executor.EvaluationError) as e:
        pm.evaluate_model_transformed(model())
    assert e.match("processed in evaluation")


def test_differently_shaped_logp():
    def model():
        yield dist.Normal("n1", np.zeros(10), np.ones(10))
        yield dist.Normal("n2", np.zeros(3), np.ones(3))

    _, state = pm.evaluate_model(model())
    state.collect_log_prob()  # this should work


def test_deterministics(model_with_deterministics):
    model, expected_deterministics, expected_ops, expected_ops_inputs = model_with_deterministics
    _, state = pm.evaluate_model(model())

    assert len(state.deterministics) == len(expected_deterministics)
    assert set(expected_deterministics) <= set(state.deterministics)
    for expected_deterministic, op, op_inputs in zip(
        expected_deterministics, expected_ops, expected_ops_inputs
    ):
        inputs = [v for k, v in state.all_values.items() if k in op_inputs]
        out = op(*inputs)
        np.testing.assert_allclose(state.deterministics[expected_deterministic], out)


def test_deterministic_with_distribution_name_fails():
    @pm.model
    def model():
        x = yield pm.Normal("x", 0, 1)
        det = yield pm.Deterministic("x", x)
        return det

    with pytest.raises(pm.flow.executor.EvaluationError):
        pm.evaluate_model(model())


def test_distribution_with_deterministic_name_fails():
    @pm.model
    def model():
        x = yield pm.Normal("x", 0, 1)
        det = yield pm.Deterministic("det", x)
        y = yield pm.Normal("det", det, 1)
        return y

    with pytest.raises(pm.flow.executor.EvaluationError):
        pm.evaluate_model(model())


def test_deterministics_in_nested_model(deterministics_in_nested_models):
    (
        model,
        expected_untransformed,
        expected_transformed,
        expected_deterministics,
        deterministic_mapping,
    ) = deterministics_in_nested_models
    _, state = pm.evaluate_model_transformed(model())
    assert set(state.untransformed_values) == expected_untransformed
    assert set(state.transformed_values) == expected_transformed
    assert set(state.deterministics) == expected_deterministics
    for deterministic, (inputs, op) in deterministic_mapping.items():
        np.testing.assert_allclose(
            state.deterministics[deterministic],
            op(*[state.untransformed_values[i] for i in inputs]),
        )
