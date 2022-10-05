from pytorch_metric_learning import miners
from lightning.base import Base


class ImageRetrievalModel(Base):
    def __init__(self, args):
        super().__init__(args)

        self.miner = miners.TripletMarginMiner(
            margin=args.margin,
            collect_stats=True,
            distance=self.distance,
            type_of_triplets="semihard",
        )

    def training_step(self, batch, batch_idx):

        x, y = batch

        output = self.forward(x)

        indices_tuple = self.get_indices_tuple(output["z_mu"], y)

        loss = self.compute_loss(output, y, indices_tuple)

        # add images to tensorboard every epoch
        if self.current_epoch == self.counter:
            if self.current_epoch < 5:
                self.log_triplets(x, indices_tuple)
            self.counter += 1

        return loss

    def get_indices_tuple(self, embeddings, labels):

        indices_tuple = self.miner(embeddings, labels, None, None)

        self.log("tuple_stats/an_dist", float(self.miner.neg_pair_dist))
        self.log("tuple_stats/ap_dist", float(self.miner.pos_pair_dist))
        self.log("tuple_stats/n_triplets", float(self.miner.num_triplets))

        return indices_tuple

    def validation_step(self, batch, batch_idx):

        x, target = batch
        output = self.forward(x)

        return {"z_mu": output["z_mu"].cpu(), "label": target.cpu()}

    def test_step(self, batch, batch_idx):

        x, target = batch
        output = self.forward(x)

        return {"z_mu": output["z_mu"].cpu(), "label": target.cpu()}
