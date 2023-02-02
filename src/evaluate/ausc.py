import torch
from sklearn.neighbors import NearestNeighbors
from utils.knn import FaissKNeighbors
import numpy as np
import matplotlib.pyplot as plt
import os
import json

def evaluate_ausc(dict, path, prefix):

    if dict["z_muDb"] is not None:
        z_muQ = dict["z_muQ"]
        z_muDb = dict["z_muDb"]
        z_sigmaQ = dict["z_sigmaQ"]
        z_sigmaDb = dict["z_sigmaDb"]

        targets = None
        pidxs = dict["pidxs"]
        same_source = False
    else:
        z_muQ = z_muDb = dict["z_muQ"]
        z_sigmaQ = z_sigmaDb = dict["z_sigmaQ"]

        targets = dict["targets"]
        pidxs = None
        same_source =True


    # plot sparsifcation curve
    # and compute ausc
    ausc, accuracies = plot_sparsification_curve(targets, pidxs, z_muQ, z_sigmaQ, z_muDb, z_sigmaDb, path, prefix, same_source)
    save_data(accuracies, path, prefix)

    #if "val" not in path and "val" not in prefix:   
        #save_data(targets=targets, 
        #        pidxs=pidxs, 
        #        z_muQ=z_muQ, 
        #        z_sigmaQ=z_sigmaQ, 
        #        z_muDb=z_muDb, 
        #        z_sigmaDb=z_sigmaDb, 
        #        path=path, 
        #        prefix=prefix,
        #        same_source=same_source)

    return ausc


def plot_sparsification_curve(targets, pidxs, z_muQ, z_sigmaQ, z_muDb, z_sigmaDb, path, prefix, same_source=True):

    # neigh = NearestNeighbors(n_neighbors=2, metric="cosine")
    z_muQ = np.ascontiguousarray(z_muQ.numpy())
    z_muDb = np.ascontiguousarray(z_muDb.numpy())

    neigh = FaissKNeighbors(k=2)
    neigh.fit(z_muDb)
    dist, idx = neigh.kneighbors(z_muQ)

    # remove itself from
    if same_source:
        dist = dist[:, 1]
        idx = idx[:, 1]
    else:
        dist = dist[:, 0]
        idx = idx[:, 0]

    if targets is not None:
        preds = targets[idx]
        correct = (preds == targets).float()
    else:
        correct = torch.from_numpy(np.stack([i in pidxs[j] for j, i in enumerate(idx)])).float()
    
    # query variance + database variance
    covar = (z_sigmaQ + z_sigmaDb[idx]).mean(dim=1)

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

    return float(ausc), accuracies

"""
def save_data(targets, pidxs, z_muQ, z_sigmaQ, z_muDb, z_sigmaDb, path, prefix, same_source):

    if same_source:
        data = {"targets" : targets.tolist(), 
                "z_mu" : z_muQ.tolist(),
                "z_sigma" : z_sigmaQ.tolist()}
    else:
        data = {"pidxs" : [p.tolist() for p in pidxs],
                "z_muQ" : z_muQ.tolist(),
                "z_sigmaQ" : z_sigmaQ.tolist(),
                "z_muDb" : z_muDb.tolist(),
                "z_sigmaDb" : z_sigmaDb.tolist()}

    os.makedirs(os.path.join(path, "figure_data"), exist_ok=True)
    with open(os.path.join(path, "figure_data", f"{prefix}sparsification_curve.json"), "w") as f:
        json.dump(data, f)
"""

def save_data(accuracies, path, prefix):
    
    accuracies = {"acc" : accuracies.tolist()}
    os.makedirs(os.path.join(path, "figure_data"), exist_ok=True)
    with open(os.path.join(path, "figure_data", f"{prefix}acc.json"), "w") as f:
        json.dump(accuracies, f)