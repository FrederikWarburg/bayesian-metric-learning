import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import os

from utils.knn import FaissKNeighbors

sns.set()


def evaluate_ece(dict, vis_path, prefix):

    if dict["z_samplesDb"] is not None:
        raise NotImplementedError

    # get samples from dict
    z_samples = dict["z_samplesQ"]
    targets = dict["targets"]

    # compute k-nn from samples
    targets, confidences, predicted = compute_knn(z_samples, targets)

    # compute calibration curve
    ece, acc, conf, bin_sizes = calibration_curves(
        targets=targets,
        confidences=confidences,
        preds=predicted,
        bins=10,
        fill_nans=True,
    )

    # plot calibration curve
    plot_calibration_curve(acc, conf, vis_path, prefix, bins=10)

    return ece


def compute_knn(samples, targets):

    N, n_samples, d = samples.shape

    pred_labels = []
    print(f"==> Computing ece predictions for {n_samples} samples")
    for i in tqdm(range(n_samples)):

        sample_i = np.ascontiguousarray(samples[:, i, :].numpy())

        # neigh = NearestNeighbors(n_neighbors=2, metric="cosine")
        neigh = FaissKNeighbors(k=2)
        neigh.fit(sample_i)
        dist, idx = neigh.kneighbors(sample_i)

        # remove itself from
        dist = dist[:, 1]
        idx = idx[:, 1]

        pred_labels.append(targets[idx])

    pred_labels = torch.stack(pred_labels)
    predicted, _ = torch.mode(pred_labels, dim=0)
    confidences = torch.mean((pred_labels == predicted).float(), dim=0)

    return targets, confidences, predicted


def calibration_curves(targets, confidences, preds, bins=10, fill_nans=False):
    targets = targets.cpu().numpy()
    confidences = confidences.cpu().numpy()
    preds = preds.cpu().numpy()

    real_probs = np.zeros((bins,))
    pred_probs = np.zeros((bins,))
    bin_sizes = np.zeros((bins,))

    _, lims = np.histogram(confidences, range=(0.0, 1.0), bins=bins)
    for i in range(bins):
        lower, upper = lims[i], lims[i + 1]
        mask = (lower <= confidences) & (confidences < upper)

        targets_in_range = targets[mask]
        preds_in_range = preds[mask]
        probs_in_range = confidences[mask]
        n_in_range = preds_in_range.shape[0]

        range_acc = (
            np.sum(targets_in_range == preds_in_range) / n_in_range
            if n_in_range > 0
            else 0
        )
        range_prob = np.sum(probs_in_range) / n_in_range if n_in_range > 0 else 0
        # range_prob = (upper + lower) / 2

        real_probs[i] = range_acc
        pred_probs[i] = range_prob
        bin_sizes[i] = n_in_range

    bin_weights = bin_sizes / np.sum(bin_sizes)
    ece = np.sum(np.abs(real_probs - pred_probs) * bin_weights)

    return ece, real_probs, pred_probs, bin_sizes
    # return ece, real_probs[bin_sizes > 0], pred_probs[bin_sizes > 0], bin_sizes


def plot_calibration_curve(acc, confidences, path, prefix, bins=10):

    fig, ax = plt.subplots()

    # Plot ECE
    ax.plot(
        np.linspace(0.05, 0.95, 10).tolist(),
        acc.tolist(),
        "-o",
        label="Calibration curve",
    )

    # Add histogram of confidences scaled between 0 and 1
    # confidences = confidences.cpu().numpy()
    ax.hist(
        confidences,
        bins=bins,
        range=(0.0, 1.0),
        density=True,
        label="Distribution of confidences",
        alpha=0.5,
    )

    # Plot identity line
    ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), "k--", label="Best fit")

    # Set axis options
    ax.set(
        xlim=[-0.1, 1.1],
        ylim=[-0.1, 1.1],
        xlabel="Confidence",
        ylabel="Accuracy",
    )

    # Add grid
    ax.grid(True, linestyle="dotted")

    # Save dir
    fig.savefig(os.path.join(path, f"{prefix}_calibration_curve.png"))
