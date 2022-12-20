import os

import numpy as np

from src.evaluate.ausc import evaluate_ausc
from src.evaluate.ece import evaluate_ece
from src.evaluate.map import mapk
from src.evaluate.ood import evaluate_ood
from src.evaluate.recall import recall


def evaluate(ranks, pidxs):

    if ranks is None:
        return np.zeros(3), np.zeros(4)

    mAPs = [mapk(ranks, pidxs, k=k) for k in [5, 10, 20]]
    recalls = recall(ranks, pidxs, ks=[1, 5, 10, 20])

    return {"map": mAPs, "recall": recalls}


def evaluate_uncertainties(dict_in, dict_ood, vis_path, prefix):

    os.makedirs(vis_path, exist_ok=True)

    metrics = {}
    if dict_ood is not None:
        # ood evaluation (auroc, auprc)
        auroc, auprc = evaluate_ood(dict_in, dict_ood, vis_path, prefix)
        metrics["auroc"] = auroc
        metrics["auprc"] = auprc

    # evaluate ece
    ece = evaluate_ece(dict_in, vis_path, prefix)
    # evaluate_ece(dict_ood, vis_path, prefix)
    metrics["ece"] = ece

    # evaluate ausc
    ausc = evaluate_ausc(dict_in, vis_path, prefix)
    # evaluate_ausc(dict_ood, vis_path, prefix)
    metrics["ausc"] = ausc

    return metrics
