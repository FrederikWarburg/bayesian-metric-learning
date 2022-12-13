from ast import Break
from dataclasses import field
import pytorch_lightning as pl
from models import configure_model, get_model_parameters
import torchvision.transforms as transforms
import torch
from pytorch_metric_learning import distances
import math
import numpy as np
from losses.loss import configure_metric_loss
from pytorch_metric_learning import distances
import wandb
import torchvision
from evaluate.utils import (
    get_pos_idx,
    get_pos_idx_place_recognition,
    remove_duplicates,
)
from evaluate.evaluate import evaluate, evaluate_uncertainties
from evaluate.ranking import compute_rank
from pytorch_metric_learning import miners
import torch
from miners.custom_miners import TripletMarginMinerPR
from miners.triplet_miners import TripletMarginMiner
import json
import os
import csv


class Base(pl.LightningModule):
    def __init__(self, args, savepath, seed):
        super().__init__()

        self.args = args
        self.model = configure_model(args)
        self.train_n_samples = args.get("train_n_samples", 1)
        self.val_n_samples = args.get("val_n_samples", 5)
        self.test_n_samples = args.get("test_n_samples", 100)

        print("==> test n_samples: ", self.test_n_samples)

        ### pytorch-metric-learning stuff ###
        if args.distance == "cosine":
            self.distance = distances.CosineSimilarity()
        elif args.distance == "euclidean":
            self.distance = distances.LpDistance()

        self.criterion = configure_metric_loss(args.loss, args.distance, args.margin)

        self.place_rec = args.dataset in ("dag", "msls")
        self.face_rec = args.dataset in ("lfw")
        self.savepath = os.path.join(savepath, "results")

        if self.place_rec:
            self.miner = TripletMarginMinerPR(
                margin=args.margin,
                collect_stats=True,
                type_of_triplets=args.type_of_triplets,
                posDistThr=self.args.posDistThr,
                negDistThr=self.args.negDistThr,
                distance=self.distance,
            )
            self.posDistThr = args.posDistThr
            self.negDistThr = args.negDistThr
        else:
            self.miner = TripletMarginMiner(
                margin=args.margin,
                collect_stats=True,
                distance=self.distance,
                type_of_triplets=args.type_of_triplets,  # [easy, hard, semihard, all]
            )

        self.counter = 0
        self.val_counter = 0
        self.inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        
        x, y = self.format_batch(batch)
        output = self.forward(x, self.train_n_samples)
        
        indices_tuple = self.get_indices_tuple(output["z_mu"], y)

        loss = self.compute_loss(output, y, indices_tuple)

        # add images to tensorboard every epoch
        if self.current_epoch == self.counter:
            if self.current_epoch < 5:
                self.log_triplets(x, indices_tuple)
            self.counter += 1

        return loss

    def training_epoch_end(self, outputs):
        torch.stack([x["loss"] for x in outputs]).mean()

    def format_batch(self, batch):
        x, y = batch
        if self.place_rec or self.face_rec:
            n = len(x)
            b, c, h, w = x[0].shape
            x = torch.stack(x).view(b * n, c, h, w)

            if self.place_rec:
                y = torch.stack(y).view(b * n, 2)
            else:
                y = torch.stack(y).view(b * n)

        return x, y

    def get_indices_tuple(self, embeddings, labels):

        if self.place_rec and self.args.split_query_database:
            b, _ = embeddings.shape

            ref_emb = embeddings[b // 2 :]
            embeddings = embeddings[: b // 2]

            ref_labels = labels[b // 2 :]
            labels = labels[: b // 2]
        else:
            ref_emb = None
            ref_labels = None

        indices_tuple = self.miner(embeddings, labels, ref_emb, ref_labels)

        self.log("tuple_stats/an_dist", float(self.miner.neg_pair_dist))
        self.log("tuple_stats/ap_dist", float(self.miner.pos_pair_dist))
        self.log("tuple_stats/n_triplets", float(self.miner.num_triplets))

        return indices_tuple

    def compute_loss(self, output, y, indices_tuple):
        raise NotImplementedError

    def forward_step(self, batch, batch_idx, dataloader_idx, n_samples=1):

        if self.place_rec:
            x, index, utm = batch
        else:
            x, target = batch

        output = self.forward(x, n_samples)

        # ensure that features are l2 normalized
        z_mu = output["z_mu"]
        z_mu = z_mu / torch.norm(z_mu, dim=-1, keepdim=True)

        o = {"z_mu": z_mu.cpu()}

        if self.place_rec:
            o["index"] = torch.stack(index).cpu()
            o["utm"] = utm.cpu()
        else:
            o["label"] = target.cpu()

        if "z_sigma" in output:
            o["z_sigma"] = output["z_sigma"].cpu()

        if "z_samples" in output:
            z_samples = output["z_samples"]
            # ensure that samples are also l2 normalize
            z_samples = z_samples / torch.norm(z_samples, dim=-1, keepdim=True)
            o["z_samples"] = z_samples.cpu()

        return o

    def validation_step(self, batch, batch_idx, dataloader_idx=1):
        return self.forward_step(
            batch, batch_idx, dataloader_idx, n_samples=self.val_n_samples
        )

    def test_step(self, batch, batch_idx, dataloader_idx=1):
        
        return self.forward_step(
            batch, batch_idx, dataloader_idx, n_samples=self.test_n_samples
        )

    def format_outputs(self, outputs):

        z_mu = torch.cat([o["z_mu"] for o in outputs])
        if not self.place_rec:
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

            pidxs = get_pos_idx_place_recognition(utmQ, utmDb, self.negDistThr)

        if z_muDb is not None and len(z_muDb) == 0:
            return

        o = {"z_muQ": z_muQ, "z_muDb": z_muDb, "pidxs": pidxs}
        if not self.place_rec:
            o["targets"] = targets

        if "z_sigma" in outputs[0]:
            z_sigma = torch.cat([o["z_sigma"] for o in outputs])
            if z_muDb is None:
                # merge dicts
                o = {**o, **{"z_sigmaQ": z_sigma, "z_sigmaDb": None}}

        if "z_samples" in outputs[0]:
            z_samples = torch.cat([o["z_samples"] for o in outputs])
            if z_muDb is None:
                # merge dicts
                o = {**o, **{"z_samplesQ": z_samples, "z_samplesDb": None}}

        return o

    def compute_metrics(self, outputs, prefix):

        if len(outputs) == 2 and "z_mu" not in outputs[0]:
            id = self.format_outputs(outputs[0])
            ood = self.format_outputs(outputs[1])
        else:
            id = self.format_outputs(outputs)
            ood = None

        if id is None:
            return None

        ranks = compute_rank(id["z_muQ"], id["z_muDb"])
        metrics = evaluate(ranks, id["pidxs"])

        if "z_sigmaQ" in id:
            uncertainty_metrics = evaluate_uncertainties(id, ood, self.savepath, prefix)
            metrics = {**metrics, **uncertainty_metrics}

        return metrics

    def validation_epoch_end(self, outputs):

        os.makedirs(
            os.path.join(self.savepath, "val", f"{self.global_step}"), exist_ok=True
        )
        metrics = self.compute_metrics(outputs, prefix=f"val/{self.global_step}/")

        if metrics is None:
            return

        for i, k in enumerate([5, 10, 20]):
            self.log("val_map/map@{}".format(k), metrics["map"][i], prog_bar=True)

        for i, k in enumerate([1, 5, 10, 20]):
            self.log("val_recall/recall@{}".format(k), metrics["recall"][i])

        for key in metrics:
            if key not in ("map", "recall"):
                self.log(f"val_metric/{key}", metrics[key])

    def test_epoch_end(self, outputs):

        metrics = self.compute_metrics(outputs, prefix="test_")

        if metrics is None:
            return

        for i, k in enumerate([5, 10, 20]):
            self.log("test_map/map@{}".format(k), metrics["map"][i], prog_bar=True)

        for i, k in enumerate([1, 5, 10, 20]):
            self.log("test_recall/recall@{}".format(k), metrics["recall"][i])

        for key in metrics:
            if key not in ("map", "recall"):
                self.log(f"test_metric/{key}", metrics[key])

        # dump to json
        with open(os.path.join(self.savepath, "metrics.json"), "w") as file:
            json.dump(metrics, file)

        # dump to csv to easy cp to google drive
        with open(os.path.join(self.savepath, "metrics.csv"), "w") as f:
            for key in ["recall", "map", "auroc", "auprc", "ausc", "ece"]:
                if key not in metrics:
                    metrics[key] = "nan"

                if key == "recall":
                    for i, k in enumerate([1, 5, 10, 20]):
                        f.write("%s\n" % (metrics[key][i]))
                elif key == "map":
                    for i, k in enumerate([5, 10, 20]):
                        f.write("%s\n" % (metrics[key][i]))
                else:
                    f.write("%s\n" % (metrics[key]))

    def configure_optimizers(self):

        parameters = get_model_parameters(self.model, self.args)

        # define optimizer
        optimizer = torch.optim.RMSprop(parameters, self.args.lr, weight_decay=10e-6)

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
