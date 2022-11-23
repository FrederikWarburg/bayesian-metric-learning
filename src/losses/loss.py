from pytorch_metric_learning import losses, distances


def configure_metric_loss(loss, distance, margin):

    if distance == "cosine":
        dist = distances.CosineSimilarity()
    elif distance == "euclidean":
        dist = distances.LpDistance()

    if loss == "triplet":
        criterion = losses.TripletMarginLoss(margin=margin, distance=dist)
    elif loss in ("contrastive", "arccos"):
        pos_margin = margin if distance == "dot" else 0
        neg_margin = 0 if distance == "dot" else margin

        criterion = losses.ContrastiveLoss(
            pos_margin=pos_margin, neg_margin=neg_margin, distance=dist
        )
    else:
        raise NotImplementedError

    return criterion
