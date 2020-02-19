"""We have notebooks placed under CI execution
as an integration test of sorts for the notebook docs.

Avoid placing long-running notebooks here,
as they will clog up the CI.
Instead, prioritize the smaller notebooks that newcomers might encounter at first.
That way, we can ensure that any breaking API changes
that may affect how newcomers interact with the library
can be caught as soon as possible.
"""

c = get_config()

c.NbConvertApp.notebooks = [
    "notebooks/baseball.ipynb",
    "notebooks/basic-usage.ipynb",
    "notebooks/rugby_analytics.ipynb",
    # will reinstate in a later PR
    #     "notebooks/radon_hierarchical.ipynb",
]
