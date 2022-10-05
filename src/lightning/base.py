import pytorch_lightning as pl
from models import configure_model, get_model_parameters
import torchvision.transforms as transforms
import torch
from pytorch_metric_learning import distances
import math
import numpy as np
from losses.loss import configure_metric_loss
from pytorch_metric_learning import distances, losses, reducers
import wandb
import torchvision
from evaluate.functionals import (
    get_pos_idx,
    evaluate,
    compute_rank,
    get_pos_idx_place_recognition,
    remove_duplicates,
)


class Base(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.model = configure_model(args)

        ### pytorch-metric-learning stuff ###
        if args.distance == "cosine":
            self.distance = distances.CosineSimilarity()
        elif args.distance == "euclidean":
            self.distance = distances.LpDistance()

        self.criterion = configure_metric_loss(args.loss, args.distance, args.margin)

        self.counter = 0
        self.val_counter = 0
        self.inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )

        self.save_hyperparameters()

    def forward(self, x):

        output = self.model(x)

        return output

    def training_step(self, batch, batch_idx):
        raise NotImplemented

    def training_epoch_end(self, outputs):
        torch.stack([x["loss"] for x in outputs]).mean()

    def compute_loss(self, output, y, indices_tuple):

        # this a hack: pytorch-metric-learning does not use the labels if indices_tuple is provided,
        # however, the shape of the labels are required to be 1D.
        place_holder = torch.zeros(y.size(0), device=y.device)

        loss = self.criterion(output["z_mu"], place_holder, indices_tuple)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
        )

        return loss

    def validation_epoch_end(self, outputs):

        z_mu = torch.cat([o["z_mu"] for o in outputs])
        if "label" in outputs[0]:
            targets = torch.cat([o["label"] for o in outputs])
            pidxs = get_pos_idx(targets)
            z_muQ = z_mu
            z_muDb = None
        else:

            index = torch.cat([o["index"] for o in outputs], dim=1)
            utm = torch.cat([o["utm"] for o in outputs])

            queries = index[1, :] == -1
            database = ~queries

            z_muQ = z_mu[queries]
            z_muDb = z_mu[database]
            utmQ = utm[queries]
            utmDb = utm[database]
            idxQ = index[0, :][queries]
            idxDb = index[1, :][database]
            z_muQ, z_muDb, utmQ, utmDb = remove_duplicates(
                z_muQ, z_muDb, utmQ, utmDb, idxQ, idxDb
            )

            pidxs = get_pos_idx_place_recognition(utmQ, utmDb, self.posDistThr)

        if z_muDb is not None and len(z_muDb) == 0:
            return

        ranks = compute_rank(z_muQ, z_muDb)
        mAPs, recalls = evaluate(ranks, pidxs)

        for i, k in enumerate([5, 10, 20]):
            self.log("val_map/map@{}".format(k), mAPs[i], prog_bar=True)

        for i, k in enumerate([1, 5, 10, 20]):
            self.log("val_recall/recall@{}".format(k), recalls[i])

    def test_epoch_end(self, outputs):

        z_mu = torch.cat([o["z_mu"] for o in outputs])
        if "label" in outputs[0]:
            targets = torch.cat([o["label"] for o in outputs])
            pidxs = get_pos_idx(targets)
            z_muQ = z_mu
            z_muDb = None
        else:
            index = torch.cat([o["index"] for o in outputs])
            utm = torch.cat([o["utm"] for o in outputs])

            queries = index[1] == -1
            database = ~queries

            z_muQ = z_mu[queries]
            z_muDb = z_mu[database]
            utmQ = utm[queries]
            utmDb = utm[database]
            idxQ = index[0][queries]
            idxDb = index[1][database]
            utmQ, utmDb, idxQ, idxDb = remove_duplicates(utmQ, utmDb, idxQ, idxDb)
            pidxs = get_pos_idx_place_recognition(utmQ, utmDb, self.posDistThr)

        if z_muDb is not None and len(z_muDb) == 0:
            return

        ranks = compute_rank(z_muQ, z_muDb)
        mAPs, recalls = evaluate(ranks, pidxs)

        for i, k in enumerate([5, 10, 20]):
            self.log("test_map/map@{}".format(k), mAPs[i], prog_bar=True)

        for i, k in enumerate([1, 5, 10, 20]):
            self.log("test_recall/recall@{}".format(k), recalls[i])

    def get_indices_tuple(self, embeddings, labels):
        raise NotImplementedError

    def configure_optimizers(self):

        parameters = get_model_parameters(self.model, self.args)

        # define optimizer
        optimizer = torch.optim.Adam(parameters, self.args.lr)

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=math.exp(-0.01)
            ),
            "name": "exponential_lr",
        }

        return [optimizer], [lr_scheduler]

    def log_triplets(self, x, indices_tuple):

        if self.args.split_query_database:
            (
                b,
                _,
                _,
                _,
            ) = x.shape
            x_ref = x[b // 2 :]
            x = x[: b // 2]
        else:
            x_ref = x

        # display triplets on tensorboard
        len_ = len(indices_tuple[0])
        index = np.random.choice(len_, min(5, len_), replace=False)
        for i, idx in enumerate(index):

            a = x[indices_tuple[0][idx]]
            p = x_ref[indices_tuple[1][idx]]
            n = x_ref[indices_tuple[2][idx]]

            if x.shape[1] == 3:
                a = self.inv_normalize(a)
                p = self.inv_normalize(p)
                n = self.inv_normalize(n)

            images = torchvision.utils.make_grid([a, p, n])
            images = wandb.Image(images.cpu().permute(1, 2, 0).numpy())
            wandb.log({f"triplets/{i}": images})
