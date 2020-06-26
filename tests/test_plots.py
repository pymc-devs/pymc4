import numpy as np
from pymc4.plots import plot_gp_dist


def test_gp_plot(tf_seed):
    """Test if the plot_gp_dist returns consistent results"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax = plot_gp_dist(ax, np.random.randn(2, 2), x=np.random.randn(2, 1), plot_samples=True)
    assert ax is not None
