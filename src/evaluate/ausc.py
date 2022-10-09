import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import os

def evaluate_ausc(dict, path, prefix):

    if dict["z_muDb"] is not None:
        raise NotImplementedError

    targets = dict["targets"]
    z_mu = dict["z_muQ"]
    z_sigma = dict["z_sigmaQ"]

    # plot sparsifcation curve
    # and compute ausc
    ausc = plot_sparsification_curve(targets, z_mu, z_sigma, path, prefix)

    return ausc

def plot_sparsification_curve(
    targets, z_mu, z_sigma, path, prefix
):

    neigh = NearestNeighbors(n_neighbors=2, metric="cosine")
    neigh.fit(z_mu)
    dist, idx = neigh.kneighbors(z_mu)

    # remove itself from
    dist = dist[:, 1]
    idx = idx[:, 1]

    preds = targets[idx]
    correct = (preds == targets).float()
    # query variance + database variance
    covar = (z_sigma + z_sigma[idx]).mean(dim=1)

    _, indices = torch.sort(-covar)

    accuracies = []
    for i in range(100):
        n = int(len(indices) * i / 100)
        accuracies.append(torch.mean(correct[indices[n:]]))

    accuracies = torch.stack(accuracies[:-1], dim=0).numpy()

    # Calculate AUSC (Area Under the Sparsification Curve)
    ausc = np.trapz(accuracies, dx=1 / len(accuracies))

    # Plot sparsification curve
    fig, ax = plt.subplots()
    ax.plot(accuracies)

    ax.set(
        xlabel="Filter Out Rate (%)",
        ylabel="Accuracy",
    )

    # Save figure
    fig.savefig(os.path.join(path, f"{prefix}sparsification_curve.png"))

    return float(ausc)
