"""
Test for hierarchical model.

Intent behind test:
*   Ensure that future API development yields the correct sample shape for
    hierarchical models.
"""
import pymc4 as pm
import tensorflow as tf
import numpy as np
import pytest
import pandas as pd

mapping = {"fortis": 0, "scandens": 1, "unknown": 2}

df = pd.read_csv(
    "https://raw.githubusercontent.com/ericmjl/bayesian-stats-modelling-tutorial/master/data/finch_beaks_2012.csv"
)
df["species_enc"] = df.apply(lambda x: mapping[x["species"]], axis=1)


@pm.model(auto_name=True)
def model():
    # SD can only be positive, therefore it is reasonable to constrain to >0
    # Likewise for betas.
    sd_hyper = pm.HalfNormal(sigma=1)
    beta_hyper = pm.HalfNormal(sigma=2)

    # Beaks cannot be of "negative" mean, therefore, HalfNormal is
    # a reasonable, constrained prior.
    mean = pm.HalfNormal(sigma=tf.fill([3], sd_hyper))
    sigma = pm.HalfNormal(sigma=tf.fill([3], beta_hyper))
    nu = pm.Exponential(lam=1 / 29.0)
    nu += 1

    # Define the likelihood distribution for the data.
    like = pm.StudentT(
        mu=tf.gather(mean, df["species_enc"]), sigma=tf.gather(sigma, df["species_enc"]), nu=nu
    )


model = model.configure()
forward_sample = model.forward_sample()


def test_hierarchical_ttest():
    assert forward_sample["like"].shape == (249,)
