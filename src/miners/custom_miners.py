import torch

# from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
# from pytorch_metric_learning.miners.base_miner import BaseTupleMiner

from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils.module_with_records_and_reducer import (
    ModuleWithRecordsAndDistance,
)


class BaseMiner(ModuleWithRecordsAndDistance):
    def mine(self, embeddings, labels, ref_emb, ref_labels):
        raise NotImplementedError

    def output_assertion(self, output):
        raise NotImplementedError

    def forward(self, embeddings, labels, ref_emb=None, ref_labels=None):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
        Does any necessary preprocessing, then does mining, and then checks the
        shape of the mining output before returning it
        """
        self.reset_stats()
        with torch.no_grad():
            # c_f.check_shapes(embeddings, labels)
            labels = c_f.to_device(labels, embeddings)
            ref_emb, ref_labels = self.set_ref_emb(
                embeddings, labels, ref_emb, ref_labels
            )
            mining_output = self.mine(embeddings, labels, ref_emb, ref_labels)
        self.output_assertion(mining_output)
        return mining_output

    def set_ref_emb(self, embeddings, labels, ref_emb, ref_labels):
        if ref_emb is not None:
            ref_labels = c_f.to_device(ref_labels, ref_emb)
        else:
            ref_emb, ref_labels = embeddings, labels
        # c_f.check_shapes(ref_emb, ref_labels)
        return ref_emb, ref_labels


class BaseTupleMiner(BaseMiner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_to_recordable_attributes(
            list_of_names=["num_pos_pairs", "num_neg_pairs", "num_triplets"],
            is_stat=True,
        )

    def output_assertion(self, output):
        """
        Args:
            output: the output of self.mine
        This asserts that the mining function is outputting
        properly formatted indices. The default is to require a tuple representing
        a,p,n indices or a1,p,a2,n indices within a batch of embeddings.
        For example, a tuple of (anchors, positives, negatives) will be
        (torch.tensor, torch.tensor, torch.tensor)
        """
        if len(output) == 3:
            self.num_triplets = len(output[0])
            assert self.num_triplets == len(output[1]) == len(output[2])
        elif len(output) == 4:
            self.num_pos_pairs = len(output[0])
            self.num_neg_pairs = len(output[2])
            assert self.num_pos_pairs == len(output[1])
            assert self.num_neg_pairs == len(output[3])
        else:
            raise BaseException


class TripletMarginMinerPR(BaseTupleMiner):
    """
    Returns triplets that violate the margin
    Args:
        margin
        type_of_triplets: options are "all", "hard", or "semihard".
                "all" means all triplets that violate the margin
                "hard" is a subset of "all", but the negative is closer to the anchor than the positive
                "semihard" is a subset of "all", but the negative is further from the anchor than the positive
            "easy" is all triplets that are not in "all"
    """

    def __init__(
        self, margin=0.2, posDistThr=10, negDistThr=25, type_of_triplets="all", **kwargs
    ):
        super().__init__(**kwargs)
        self.margin = margin
        self.posDistThr = posDistThr
        self.negDistThr = negDistThr
        self.type_of_triplets = type_of_triplets
        self.add_to_recordable_attributes(list_of_names=["margin"], is_stat=False)
        self.add_to_recordable_attributes(
            list_of_names=["avg_triplet_margin", "pos_pair_dist", "neg_pair_dist"],
            is_stat=True,
        )

    def mine(self, embeddings, labels, ref_emb, ref_labels):

        anchor_idx, positive_idx, negative_idx = get_all_triplets_indices(
            labels, ref_labels, self.posDistThr, self.negDistThr
        )

        mat = self.distance(embeddings, ref_emb)
        ap_dist = mat[anchor_idx, positive_idx]
        an_dist = mat[anchor_idx, negative_idx]
        triplet_margin = (
            ap_dist - an_dist if self.distance.is_inverted else an_dist - ap_dist
        )

        if self.type_of_triplets == "easy":
            threshold_condition = triplet_margin > self.margin
        else:
            # check the triplet that violates the triplet constraint
            threshold_condition = triplet_margin <= self.margin
            if self.type_of_triplets == "hard":
                threshold_condition &= triplet_margin <= 0
            elif self.type_of_triplets == "semihard":
                threshold_condition &= triplet_margin > 0

        self.set_stats(ap_dist, an_dist, triplet_margin)

        # Triplet that violates the triplet constraint
        if self.type_of_triplets != "none":
            anchor_idx = anchor_idx[threshold_condition]
            positive_idx = positive_idx[threshold_condition]
            negative_idx = negative_idx[threshold_condition]

        return (
            anchor_idx,
            positive_idx,
            negative_idx,
        )

    def set_stats(self, ap_dist, an_dist, triplet_margin):
        if self.collect_stats:
            with torch.no_grad():
                self.pos_pair_dist = torch.mean(ap_dist).item()
                self.neg_pair_dist = torch.mean(an_dist).item()
                self.avg_triplet_margin = torch.mean(triplet_margin).item()


def get_all_pairs_indices(labels, ref_labels=None, posDistThr=10, negDistThr=25):
    """
    Given a tensor of labels, this will return 4 tensors.
    The first 2 tensors are the indices which form all positive pairs
    The second 2 tensors are the indices which form all negative pairs
    """
    if ref_labels is None:
        ref_labels = labels
    labels1 = labels
    labels2 = ref_labels

    dist = torch.cdist(labels1, labels2, p=2)

    matches = (dist < posDistThr).byte()
    diffs = (dist > negDistThr).byte()

    if ref_labels is labels:
        matches.fill_diagonal_(0)
    a1_idx, p_idx = torch.where(matches)
    a2_idx, n_idx = torch.where(diffs)

    return a1_idx, p_idx, a2_idx, n_idx


def get_all_triplets_indices(labels, ref_labels=None, posDistThr=10, negDistThr=25):

    if ref_labels is None:
        ref_labels = labels
    labels1 = labels
    labels2 = ref_labels

    dist = torch.cdist(labels1, labels2, p=2)

    matches = (dist < posDistThr).byte()
    diffs = (dist > negDistThr).byte()

    if ref_labels is labels:
        matches.fill_diagonal_(0)

    triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
    return torch.where(triplets)
