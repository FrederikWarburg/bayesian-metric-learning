import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchmetrics

from src.evaluate.visualize_samples import plot_samples


def evaluate_ood(dict_in, dict_ood, vis_path, prefix):

    if dict_ood is None:
        return None, None

    if dict_in["z_muDb"] is not None:
        raise NotADirectoryError

    id_z_mu = dict_in["z_muQ"]
    id_sigma = dict_in["z_sigmaQ"]
    ood_z_mu = dict_ood["z_muQ"]
    ood_sigma = dict_ood["z_sigmaQ"]

    # plot histrograms
    plot_ood(
        id_z_mu,
        id_sigma,
        ood_z_mu,
        ood_sigma,
        vis_path,
        prefix,
    )

    id_sigma = np.reshape(id_sigma, (id_sigma.shape[0], -1))
    ood_sigma = np.reshape(ood_sigma, (ood_sigma.shape[0], -1))

    id_sigma, ood_sigma = id_sigma.sum(axis=1), ood_sigma.sum(axis=1)

    pred = np.concatenate([id_sigma, ood_sigma])
    target = np.concatenate([[0] * len(id_sigma), [1] * len(ood_sigma)])

    # compute auroc
    auroc_score = compute_auroc(pred, target)

    # compute auprc
    auprc_score = compute_auprc(pred, target)

    # plot roc
    plot_roc(pred, target, vis_path, prefix)

    # plot prc
    plot_prc(pred, target, vis_path, prefix)

    if "val" not in vis_path and "val" not in prefix:
        save_data(pred, target, vis_path, prefix)

    return float(auroc_score.numpy()), float(auprc_score.numpy())


def plot_histogram(sigma2, ax=None, color="b", label=None):
    if ax is None:
        _, ax = plt.subplots()

    mean_sigma_sq = np.mean(sigma2.numpy(), axis=1)

    sns.kdeplot(mean_sigma_sq, ax=ax, color=color, label=label)
    ax.set(xlabel="Variance")


def plot_ood(mu_id, var_id, mu_ood, var_ood, vis_path, prefix):

    fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
    plot_samples(mu_id, var_id, limit=100, color="red", label="ID", ax=ax[0])
    plot_histogram(var_id, color="red", ax=ax[1])
    plot_samples(mu_ood, var_ood, limit=100, color="blue", label="OOD", ax=ax[0])
    plot_histogram(var_ood, color="blue", ax=ax[1])
    ax[0].legend()
    fig.savefig(os.path.join(vis_path, f"{prefix}ood_comparison.png"))
    return fig, ax


def compute_auroc(pred, target):

    # compute auroc
    auroc = torchmetrics.AUROC(num_classes=1)
    auroc_score = auroc(torch.tensor(pred).unsqueeze(1), torch.tensor(target).unsqueeze(1))

    return auroc_score


def compute_auprc(pred, target):

    # plot precision recall curve
    pr_curve = torchmetrics.PrecisionRecallCurve(pos_label=1)
    precision, recall, thresholds = pr_curve(torch.tensor(pred).unsqueeze(1), torch.tensor(target).unsqueeze(1))

    # compute auprc (area under precission recall curve)
    auc = torchmetrics.AUC(reorder=True)
    auprc_score = auc(recall, precision)

    return auprc_score


def plot_prc(pred, target, vis_path, prefix):

    # plot precision recall curve
    pr_curve = torchmetrics.PrecisionRecallCurve(pos_label=1)
    precision, recall, thresholds = pr_curve(torch.tensor(pred).unsqueeze(1), torch.tensor(target).unsqueeze(1))

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    fig.savefig(os.path.join(vis_path, f"{prefix}ood_precision_recall_curve.png"))
    plt.close()
    plt.cla()
    plt.clf()


def plot_roc(pred, target, vis_path, prefix):

    # plot roc curve
    roc = torchmetrics.ROC(num_classes=1)
    fpr, tpr, thresholds = roc(torch.tensor(pred).unsqueeze(1), torch.tensor(target).unsqueeze(1))

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    fig.savefig(os.path.join(vis_path, f"{prefix}ood_roc_curve.png"))
    plt.close()
    plt.cla()
    plt.clf()


def save_data(pred, target, path, prefix):

    data = {"pred": pred.tolist(), "target": target.tolist()}

    os.makedirs(os.path.join(path, "figure_data"), exist_ok=True)
    with open(os.path.join(path, "figure_data", f"{prefix}ood_curves.json"), "w") as f:
        json.dump(data, f)
