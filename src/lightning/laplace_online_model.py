from lightning.base import Base
import torch
import torch.nn as nn
import os
import sys

sys.path.append("../stochman")
from stochman import nnj
from stochman import ContrastiveHessianCalculator, ArccosHessianCalculator
from stochman.laplace import DiagLaplace
from stochman.utils import convert_to_stochman
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm
import wandb

from miners.custom_miners import TripletMarginMinerPR
from miners.triplet_miners import TripletMarginMiner


hessian_calculators = {
    "contrastive_pos": ContrastiveHessianCalculator,
    "contrastive_fix": ContrastiveHessianCalculator,
    "contrastive_full": ContrastiveHessianCalculator,
    "arccos_pos": ArccosHessianCalculator,
    # "arccos_fix": ArccosHessianCalculator,
    "arccos_full": ArccosHessianCalculator,
}


class LaplaceOnlineModel(Base):
    def __init__(self, args, savepath, seed):
        super().__init__(args, savepath, seed)

        self.max_pairs = args.max_pairs

        # transfer part of model to stochman
        self.model.linear = convert_to_stochman(self.model.linear)

        loss_func = f"{args.loss}_{args.loss_approx}"
        self.hessian_calculator = hessian_calculators[loss_func](
            wrt="weight", loss_func=loss_func, shape="diagonal", speed="half"
        )

        self.laplace = DiagLaplace()

        self.dataset_size = args.dataset_size
        hessian = self.laplace.init_hessian(
            self.dataset_size, self.model.linear, "cuda:0"
        )
        self.register_buffer("hessian", hessian)

        self.hessian_memory_factor = args.hessian_memory_factor

        # hessian miners
        if self.place_rec:
            self.hessian_miner = TripletMarginMinerPR(
                margin=args.margin,
                collect_stats=True,
                type_of_triplets=args.type_of_triplets_hessian,
                posDistThr=self.args.posDistThr,
                negDistThr=self.args.negDistThr,
                distance=self.distance,
            )
        else:
            self.hessian_miner = TripletMarginMiner(
                margin=args.margin,
                collect_stats=True,
                distance=self.distance,
                type_of_triplets=args.type_of_triplets_hessian,  # [easy, hard, semihard, all]
            )

    def training_step(self, batch, batch_idx):

        x, y = self.format_batch(batch)

        x = self.model.backbone(x)
        if hasattr(self.model, "pool"):
            x = self.model.pool(x)

        # get mean and std of posterior
        sigma_q = self.laplace.posterior_scale(torch.relu(self.hessian))
        mu_q = parameters_to_vector(self.model.linear.parameters()).unsqueeze(1)

        # init variables to store running sums
        loss = 0
        hessian = None

        # draw samples from the nn (sample nn)
        samples = self.laplace.sample(mu_q, sigma_q, self.train_n_samples)
        for net_sample in samples:

            # replace the network parameters with the sampled parameters
            vector_to_parameters(net_sample, self.model.linear.parameters())

            z = self.model.linear(x)

            indices_tuple = self.get_indices_tuple(z, y)

            loss += self.compute_loss(z, y, indices_tuple)

            with torch.inference_mode():

                hessian_indices_tuple = self.get_hessian_indices_tuple(z, y)

                # randomly choose 5000 pairs if more than 5000 pairs available.
                # TODO: decide what to do. What pairs should we use to compute the hessian over?
                # does it matter? What experiments should we run to get a better idea?
                if len(hessian_indices_tuple[0]) > self.max_pairs:
                    idx = torch.randperm(hessian_indices_tuple[0].size(0))[: self.max_pairs]
                    hessian_indices_tuple = (
                        hessian_indices_tuple[0][idx],
                        hessian_indices_tuple[1][idx],
                        hessian_indices_tuple[2][idx],
                    )

                h_s = self.hessian_calculator.compute_hessian(
                    x.detach(), self.model.linear, hessian_indices_tuple
                )
                h_s = self.laplace.scale(h_s, x.shape[0], self.dataset_size)

            if hessian is None:
                hessian = h_s
            else:
                hessian += hessian

        # reset the network parameters with the mean parameter (MAP estimate parameters)
        vector_to_parameters(mu_q, self.model.linear.parameters())
        loss = loss / self.train_n_samples
        hessian = hessian / self.train_n_samples

        self.hessian = self.hessian_memory_factor * self.hessian + hessian

        # add images to tensorboard every epoch
        if self.current_epoch == self.counter:
            if self.current_epoch < 5:
                self.log_triplets(x, indices_tuple)
            self.counter += 1

        # log hessian and sigma_q
        wandb.log({"extra/hessian" : self.hessian.sum()})
        wandb.log({"extra/sigma_q" : sigma_q.sum()})

        return loss

    def forward(self, x, n_samples=1):

        x = self.model.backbone(x)
        if hasattr(self.model, "pool"):
            x = self.model.pool(x)

        # get mean and std of posterior
        mu_q = parameters_to_vector(self.model.linear.parameters()).unsqueeze(1)
        self.hessian = torch.relu(self.hessian)
        sigma_q = self.laplace.posterior_scale(self.hessian)

        # draw samples
        samples = self.laplace.sample(mu_q, sigma_q, n_samples)

        # forward n times
        zs = []
        for net_sample in samples:

            # replace the network parameters with the sampled parameters
            vector_to_parameters(net_sample, self.model.linear.parameters())

            z = self.model.linear(x)

            zs.append(z)

        zs = torch.stack(zs)

        # compute statistics
        z_mu = zs.mean(dim=0)
        z_sigma = zs.std(dim=0)

        # put mean parameters back
        vector_to_parameters(mu_q, self.model.linear.parameters())

        return {"z_mu": z_mu, "z_sigma": z_sigma, "z_samples": zs.permute(1, 0, 2)}

    def compute_loss(self, z, y, indices_tuple):

        # this a hack: pytorch-metric-learning does not use the labels if indices_tuple is provided,
        # however, the shape of the labels are required to be 1D.
        place_holder = torch.zeros(y.size(0), device=y.device)

        if self.args.loss == "arccos":
            z = z / torch.norm(z, dim=-1, keepdim=True)

        loss = self.criterion(z, place_holder, indices_tuple)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
        )

        return loss

    def get_hessian_indices_tuple(self, embeddings, labels):

        if self.place_rec and self.args.split_query_database:
            b, _ = embeddings.shape

            ref_emb = embeddings[b // 2 :]
            embeddings = embeddings[: b // 2]

            ref_labels = labels[b // 2 :]
            labels = labels[: b // 2]
        else:
            ref_emb = None
            ref_labels = None

        indices_tuple = self.hessian_miner(embeddings, labels, ref_emb, ref_labels)

        self.log("hessian_tuple_stats/an_dist", float(self.hessian_miner.neg_pair_dist))
        self.log("hessian_tuple_stats/ap_dist", float(self.hessian_miner.pos_pair_dist))
        self.log("hessian_tuple_stats/n_triplets", float(self.hessian_miner.num_triplets))

        return indices_tuple