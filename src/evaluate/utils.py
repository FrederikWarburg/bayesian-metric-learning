import numpy as np
from sklearn.neighbors import NearestNeighbors


def get_pos_idx(target):

    classes = np.unique(target)
    idx = {}
    for c in classes:
        idx[f"{c}"] = {"pos": np.where(target == c)[0]}

    pos_idx = []
    for i in range(len(target)):
        key = f"{target[i].data}"

        pidx = idx[key]["pos"]
        pidx = pidx[pidx != i]  # remove self

        pos_idx.append(pidx)

    return pos_idx


def get_pos_idx_place_recognition(utmQ, utmDb, posDistThr):

    if len(utmDb) == 0 or len(utmQ) == 0:
        return []

    neigh = NearestNeighbors(algorithm="brute")
    neigh.fit(utmDb)
    _, pidxs = neigh.radius_neighbors(utmQ, posDistThr)

    return pidxs


def remove_duplicates(z_muQ, z_muDb, utmQ, utmDb, idxQ, idxDb):

    # remove duplicates
    q_keys = {}
    for i, val in enumerate(idxQ.numpy()):
        if val not in q_keys:
            q_keys[val] = i

    db_keys = {}
    for i, val in enumerate(idxDb.numpy()):
        if val not in db_keys:
            db_keys[val] = i

    q_index = np.array([q_keys[i] for i in q_keys.values()], dtype=int)
    db_index = np.array([db_keys[i] for i in db_keys.values()], dtype=int)

    z_muQ = z_muQ[q_index]
    z_muDb = z_muDb[db_index]
    utmQ = utmQ[q_index]
    utmDb = utmDb[db_index]

    return z_muQ, z_muDb, utmQ, utmDb, q_index, db_index
