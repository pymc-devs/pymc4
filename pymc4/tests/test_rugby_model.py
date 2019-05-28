import pymc4 as pm4
import tensorflow as tf
from tensorflow_probability import bijectors
import numpy as np

import pandas as pd

RUGBY_DATA_URL = (
    "https://raw.githubusercontent.com/pymc-devs/pymc3/master/pymc3/examples/data/rugby.csv"
)
tf.random.set_seed(1234)
df_all = pd.read_csv(RUGBY_DATA_URL)

df = df_all[["home_team", "away_team", "home_score", "away_score"]]

teams = df.home_team.unique()
teams = pd.DataFrame(teams, columns=["team"])
teams["i"] = teams.index

df = pd.merge(df, teams, left_on="home_team", right_on="team", how="left")
df = df.rename(columns={"i": "i_home"}).drop("team", 1)
df = pd.merge(df, teams, left_on="away_team", right_on="team", how="left")
df = df.rename(columns={"i": "i_away"}).drop("team", 1)

observed_home_goals = df.home_score.values
observed_away_goals = df.away_score.values

home_team = df.i_home.values
away_team = df.i_away.values

num_teams = len(df.i_home.drop_duplicates())
num_games = len(home_team)

g = df.groupby("i_away")
att_starting_points = np.log(g.away_score.mean())
g = df.groupby("i_home")
def_starting_points = -np.log(g.away_score.mean())

# Logp calculation for linear regression
@pm4.model(auto_name=True)
def rugby():
    # Define priors
    home = pm4.Normal(mu=0, sigma=2)
    sd_att = pm4.HalfNormal(sigma=2.5)
    sd_def = pm4.HalfNormal(sigma=2.5)
    intercept = pm4.Normal(mu=0, sigma=2)

    # team-specific model parameters
    atts_star = pm4.Normal(mu=0, sigma=tf.fill([6], sd_att))
    defs_star = pm4.Normal(mu=0, sigma=tf.fill([6], sd_def))

    atts = atts_star
    defs = defs_star
    print(defs)
    home_theta = tf.math.exp(
        intercept + home + tf.gather(atts, home_team) + tf.gather(defs, away_team)
    )
    away_theta = tf.math.exp(intercept + tf.gather(atts, away_team) + tf.gather(defs, home_team))
    print(home_theta)
    # likelihood of observed data
    home_points = pm4.Poisson(mu=home_theta)
    away_points = pm4.Poisson(mu=away_theta)


def test_forward_sample_rugby():
    model = rugby.configure()

    forward_sample = model.forward_sample()
    assert len(forward_sample["away_points"]) == 60
    assert len(forward_sample["home_points"]) == 60
    assert tf.math.equal(forward_sample["home"], 1.6069661)
