import numpy as np


def recall(ranks, pidx, ks):

    recall_at_k = np.zeros(len(ks))
    for qidx in range(ranks.shape[0]):

        for i, k in enumerate(ks):
            if np.sum(np.in1d(ranks[qidx, :k], pidx[qidx])) > 0:
                recall_at_k[i:] += 1
                break

    recall_at_k /= ranks.shape[0]

    return recall_at_k.tolist()
