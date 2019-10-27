"""
Plotting utility functions for PyMC4.

Plotting functions are delegated to the ArviZ library, a general purpose library
for exploratory analysis of Bayesian models. See
https://arviz-devs.github.io/arviz/ for details on plots.
"""
import functools
import sys
import arviz as az


# Access to arviz plots: base plots provided by arviz
for plot in az.plots.__all__:
    setattr(sys.modules[__name__], plot, (getattr(az.plots, plot)))


# Use compact traceplot by default
@functools.wraps(az.plot_trace)
def plot_trace(*args, **kwargs):
    try:
        kwargs.setdefault("compact", True)
        return az.plot_trace(*args, **kwargs)
    except TypeError:
        kwargs.pop("compact")
        return az.plot_trace(*args, **kwargs)


# Additional argument mapping for plot_compare
@functools.wraps(az.plot_compare)
def plot_compare(*args, **kwargs):
    if "comp_df" in kwargs:
        comp_df = kwargs["comp_df"].copy()
    else:
        args = list(args)
        comp_df = args[0].copy()

    if "WAIC" in comp_df.columns:
        comp_df = comp_df.rename(
            index=str,
            columns={
                "WAIC": "waic",
                "pWAIC": "p_waic",
                "dWAIC": "d_waic",
                "SE": "se",
                "dSE": "dse",
                "var_warn": "warning",
            },
        )
    elif "LOO" in comp_df.columns:
        comp_df = comp_df.rename(
            index=str,
            columns={
                "LOO": "loo",
                "pLOO": "p_loo",
                "dLOO": "d_loo",
                "SE": "se",
                "dSE": "dse",
                "shape_warn": "warning",
            },
        )

    if "comp_df" in kwargs:
        kwargs["comp_df"] = comp_df
    else:
        args[0] = comp_df

    return az.plot_compare(*args, **kwargs)


__all__ = az.plots.__all__
