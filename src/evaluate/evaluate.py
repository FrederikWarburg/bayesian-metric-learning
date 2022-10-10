from evaluate.visualize_samples import ood_visualisations
import numpy as np
from evaluate.map import mapk
from evaluate.recall import recall
from evaluate.ood import evaluate_ood
from evaluate.ece import evaluate_ece
from evaluate.ausc import evaluate_ausc
import os


def evaluate(ranks, pidxs):

    if ranks is None:
        return np.zeros(3), np.zeros(4)

    mAPs = [mapk(ranks, pidxs, k=k) for k in [5, 10, 20]]
    recalls = recall(ranks, pidxs, ks=[1, 5, 10, 20])

    return {"map": mAPs, "recall": recalls}


def evaluate_uncertainties(dict_in, dict_ood, vis_path, prefix):

    os.makedirs(vis_path, exist_ok=True)

    # ood evaluation (auroc, auprc)
    auroc, auprc = evaluate_ood(dict_in, dict_ood, vis_path, prefix)

    # evaluate ece
    ece = evaluate_ece(dict_in, vis_path, prefix)
    # evaluate_ece(dict_ood, vis_path, prefix)

    # evaluate ausc
    ausc = evaluate_ausc(dict_in, vis_path, prefix)
    # evaluate_ausc(dict_ood, vis_path, prefix)

    return {"auroc": auroc, "auprc": auprc, "ece": ece, "ausc": ausc}
