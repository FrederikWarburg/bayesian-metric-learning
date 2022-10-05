import torch
from miners.custom_miners import TripletMarginMiner
from lightning.base import Base


class PlaceRecognitionModel(Base):
    def __init__(self, args):
        super().__init__(args)

        self.miner = TripletMarginMiner(
            margin=args.margin,
            collect_stats=True,
            type_of_triplets=args.miner,
            posDistThr=self.args.posDistThr,
            negDistThr=self.args.negDistThr,
            distance=self.distance,
        )

        self.posDistThr = args.posDistThr

    def training_step(self, batch, batch_idx):

        x, y = batch
        n = len(x)
        b, c, h, w = x[0].shape
        x = torch.stack(x).view(b * n, c, h, w)
        y = torch.stack(y).view(b * n, 2)

        output = self(x)

        indices_tuple = self.get_indices_tuple(output["z_mu"], y)

        loss = self.compute_loss(output, y, indices_tuple)

        # add images to tensorboard every epoch
        if self.current_epoch == self.counter:
            self.log_triplets(x, indices_tuple)
            self.counter += 1

        return loss

    def get_indices_tuple(self, embeddings, labels):

        if self.args.split_query_database:
            b, _ = embeddings.shape

            ref_emb = embeddings[b // 2 :]
            embeddings = embeddings[: b // 2]

            ref_labels = labels[b // 2 :]
            labels = labels[: b // 2]
        else:
            ref_emb = None
            ref_labels = None

        indices_tuple = self.miner(embeddings, labels, ref_emb, ref_labels)

        self.log("tuple_stats/an_dist", self.miner.neg_pair_dist)
        self.log("tuple_stats/ap_dist", self.miner.pos_pair_dist)
        self.log("tuple_stats/n_triplets", self.miner.num_triplets)

        return indices_tuple

    def validation_step(self, batch, batch_idx):

        x, index, utm = batch
        output = self.forward(x)

        return {
            "z_mu": output["z_mu"].cpu(),
            "index": torch.stack(index).cpu(),
            "utm": utm.cpu(),
        }

    def test_step(self, batch, batch_idx):

        x, index, utm = batch
        output = self.forward(x)

        return {
            "z_mu": output["z_mu"].cpu(),
            "index": torch.stack(index).cpu(),
            "utm": utm.cpu(),
        }
