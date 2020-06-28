import matplotlib.pyplot as plt
import numpy as np

def plot_samples(x, batched_samples, labels, names, ylim=None):
    if not isinstance(batched_samples, np.ndarray):
        batched_samples = np.asarray(batched_samples)
    n_samples = batched_samples.shape[0]
    if ylim is not None:
        ymin, ymax = ylim
    else:
        ymin, ymax = batched_samples.min()-0.2, batched_samples.max()+0.2
    fig, ax = plt.subplots(n_samples, 1, figsize=(14, n_samples*3))
    if isinstance(labels, (list, tuple)):
        labels = [np.asarray(label) for label in labels]
    else:
        labels = np.asarray(labels)
    for i in range(len(ax)):
        samples = batched_samples[i]
        axi = ax[i]
        if isinstance(labels, (list, tuple)):
            lab = names[0] + "=" + str(labels[0][i])
            for l, name in zip(labels[1:], names[1:]):
                lab += ", " + name + "=" + str(l[i])
        else:
            lab = names + "=" + str(labels[i])
        for sample in samples:
            axi.plot(x, sample, label=lab)
            axi.set_ylim(ymin=ymin, ymax=ymax)
        axi.set_title(lab)
    plt.show()

def plot_cov_matrix(k, X, labels, names, vlim=None, cmap="inferno", interpolation="none"):
    cov = k(X, X)
    cov = np.asarray(cov)
    if vlim is not None:
        vmin, vmax = vlim
    else:
        vmin, vmax = cov.min(), cov.max()
    if isinstance(labels, (list, tuple)):
        labels = [np.asarray(label) for label in labels]
        n_samples = len(labels[0])
    else:
        labels = np.asarray(labels)
        n_samples = 1
    fig, ax = plt.subplots(1, n_samples, figsize=(5*n_samples, 4))
    if not isinstance(ax, np.ndarray): ax = np.asarray([ax])
    for i in range(ax.shape[0]):
        axi = ax[i]
        if isinstance(labels, (list, tuple)):
            lab = names[0] + "=" + str(labels[0][i])
            for l, name in zip(labels[1:], names[1:]):
                lab += ", " + name + "=" + str(l[i])
        else:
            lab = names + "=" + str(labels[i])
        m = axi.imshow(cov[i], cmap=cmap, interpolation=interpolation)
        m.set_clim(vmin=vmin, vmax=vmax)
        plt.colorbar(m, ax=axi)
        axi.grid(False)
        axi.set_title(lab)
    plt.show()
