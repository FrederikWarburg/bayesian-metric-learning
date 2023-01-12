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
    "contrastive": ContrastiveHessianCalculator,
    "arccos": ArccosHessianCalculator,
}


class LaplaceOnlineModel(Base):
    def __init__(self, args, savepath, seed):
        super().__init__(args, savepath, seed)

        self.max_pairs = args.max_pairs

        # if arccos, then remove normalization layer from model
        if self.args.loss == "arccos":
            self.model.linear = convert_to_stochman(self.model.linear[:-1])
        else:
            self.model.linear = convert_to_stochman(self.model.linear)

        self.hessian_calculator = hessian_calculators[args.loss](
            wrt="weight", 
            shape="diagonal", 
            speed="half", 
            method=args.loss_approx,
        )

        self.laplace = DiagLaplace()

        self.dataset_size = args.dataset_size
        hessian = self.laplace.init_hessian(
            args.get("init_hessian", self.dataset_size), self.model.linear, "cuda:0"
        )
        self.register_buffer("hessian", hessian)
        self.prior_prec = torch.tensor(1, device="cuda:0")

        #self.n_step_without_hessian_update = args.get("n_step_without_hessian_update", 0)
        #self.n_step_to_introduce_hessian = args.get("n_step_to_introduce_hessian", 0)
        #self.hessian_step_counter = 0
        self.hessian_memory_factor = args.hessian_memory_factor
        
        # hessian miners
        if self.place_rec:
            self.hessian_miner = TripletMarginMinerPR(
                margin=args.margin,
                collect_stats=True,
                type_of_triplets=args.get("type_of_triplets_hessian", "all"),
                posDistThr=args.get("posDistThr", 10),
                negDistThr=args.get("negDistThr", 25),
                distance=self.distance,
            )
        else:
            self.hessian_miner = TripletMarginMiner(
                margin=args.margin,
                collect_stats=True,
                distance=self.distance,
                type_of_triplets=args.get("type_of_triplets_hessian", "all"),  # [easy, hard, semihard, all]
            )

    """
    def on_train_batch_start(self, batch, batch_idx):
        
        if self.hessian_step_counter < self.n_step_without_hessian_update:
            # do not update the hessian
            self.hessian_memory_factor = 1

        elif self.hessian_step_counter < self.n_step_to_introduce_hessian:
            # slowly decrease memory factor
            steps_to_introduce_hessian = self.n_step_to_introduce_hessian - self.n_step_without_hessian_update
            step = self.hessian_step_counter - self.n_step_without_hessian_update 
            weight = 1 - (steps_to_introduce_hessian - step) / steps_to_introduce_hessian 
            self.hessian_memory_factor = weight * self.args.hessian_memory_factor + (1 - weight) * 1
        
        self.hessian_step_counter += 1
        
        wandb.log({"extra/hessian_memory_factor": self.hessian_memory_factor})
        wandb.log({"extra/hessian_step_counter": self.hessian_step_counter})
    """

    def training_step(self, batch, batch_idx):

        im, y = self.format_batch(batch)

        x = self.model.backbone(im)
        if hasattr(self.model, "pool"):
            x = self.model.pool(x)

        # get mean and std of posterior
        sigma_q = self.laplace.posterior_scale(torch.relu(self.hessian))
        mu_q = parameters_to_vector(self.model.linear.parameters()).unsqueeze(1)

        # init variables to store running sums
        loss = 0
        hessian = 0

        # draw samples from the nn (sample nn)
        samples = self.laplace.sample(mu_q, sigma_q, self.train_n_samples)
        for net_sample in samples:

            # replace the network parameters with the sampled parameters
            vector_to_parameters(net_sample, self.model.linear.parameters())

            z = self.model.linear(x)

            # ensure that we are on unit sphere
            z = z / z.norm(dim=-1, keepdim=True)

            indices_tuple = self.get_indices_tuple(z, y)

            loss += self.compute_loss(z, y, indices_tuple)

            with torch.inference_mode():

                hessian_indices_tuple = self.get_hessian_indices_tuple(z, y)

                # randomly choose 5000 pairs if more than 5000 pairs available.
                # TODO: decide what to do. What pairs should we use to compute the hessian over?
                # does it matter? What experiments should we run to get a better idea?
                n_triplets = len(hessian_indices_tuple[0])
                if n_triplets > self.max_pairs:
                    idx = torch.randperm(hessian_indices_tuple[0].size(0))[: self.max_pairs]
                    hessian_indices_tuple = (
                        hessian_indices_tuple[0][idx],
                        hessian_indices_tuple[1][idx],
                        hessian_indices_tuple[2][idx],
                    )

                h_s = self.hessian_calculator.compute_hessian(
                    x.detach(), self.model.linear, hessian_indices_tuple
                )
                h_s = self.laplace.scale(h_s, min(n_triplets, self.max_pairs), self.dataset_size**2)
                hessian += h_s

        # reset the network parameters with the mean parameter (MAP estimate parameters)
        vector_to_parameters(mu_q, self.model.linear.parameters())
        loss = loss / self.train_n_samples
        hessian = hessian / self.train_n_samples

        self.hessian = self.hessian_memory_factor * self.hessian + hessian

        # add images to tensorboard every epoch
        if self.current_epoch == self.counter:
            if self.current_epoch < 5:
                self.log_triplets(im, indices_tuple)
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

        # forward n times
        zs = []
        for i in range(n_samples):

            # use sample i that was generated in beginning of evaluation
            net_sample = self.nn_weight_samples[i]

            # replace the network parameters with the sampled parameters
            vector_to_parameters(net_sample, self.model.linear.parameters())

            z = self.model.linear(x)

            # ensure that we are on unit sphere
            z = z / z.norm(dim=-1, keepdim=True)

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

    def sample(self, n_samples):

        if not hasattr(self, "hessian"):
            print("==> no hessian found!")
            sys.exit()

        # get mean and std of posterior
        mu_q = parameters_to_vector(self.model.linear.parameters()).unsqueeze(1)
        sigma_q = self.laplace.posterior_scale(torch.relu(self.hessian))

        # draw samples
        self.nn_weight_samples = self.laplace.sample(mu_q, sigma_q, n_samples)

    def on_validation_epoch_start(self):
        self.sample(self.val_n_samples)

    def on_test_epoch_start(self):
        self.sample(self.test_n_samples)