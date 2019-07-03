"""
TODO: Tests for PyMC4 auto naming.
"""
# AST parsing only works in non-nested functions, thus place things into module namespace
import pytest
import pymc4 as pm


def create_rvs(**kwargs):
    # AST parsers should just add name kwargs to pymc4 RVs
    assert "name" not in kwargs.keys()
    return pm.Normal(0, 1, name="inside_function")


def autoname_test():
    inferred_w_module = pm.Normal(0, 1)
    inferred_wo_module = Normal(0, 1)
    inferred_full_path = pm.random_variables.Normal(0, 1)
    pm.Normal(0, 1, name="supplied_name")
    dummy = pm.Normal(0, 1, name="overwrite_name")
    dummy + inferred_w_module
    x = create_rvs()


# TODO: make autoname work, to not test for now
def no_test_auto_name():
    model = autoname_test.configure()

    rv_names = [rv.name for rv in model._forward_context.vars]
    expected_rv_names = [
        "inferred_w_module",
        "inferred_wo_module",
        "inferred_full_path",
        "supplied_name",
        "overwrite_name",
        "inside_function",
    ]

    assert rv_names == expected_rv_names
